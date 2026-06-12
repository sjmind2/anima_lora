/**
 * came_cuda_batched.cu — Batched CUDA kernels for CAME optimizer.
 *
 * Batched variants of the 8 single-param kernels. Processes B parameters
 * of the same shape (R, C) in a single set of kernel launches.
 *
 * Key difference from single-param kernels: all tensors have a leading
 * batch dimension B. Pointer arithmetic adds b*R*C (or b*C, b*R) offsets.
 *
 *   P0: batched_prepare_param             — dtype conversion + weight decay + scalar_buf zero
 *   S0: batched_scale2                    — column state pre-multiply (B*C elements)
 *   F1: batched_compute_sq_ema            — grad^2+eps0, row/col EMA (grid dim3(B,R))
 *   F2: batched_first_approx_and_norm     — first approx, L2 norm (grid dim3(B,R))
 *   F3: batched_clip_ema_residual_ema     — RMS clip, exp_avg EMA, residual EMA (grid dim3(B,R))
 *   F4: batched_second_approx_and_apply   — second approx, param update + dtype writeback (grid dim3(B,R))
 *   U1: batched_unfactored_sq_rsqrt_norm  — exp_avg_sq EMA, rsqrt*grad, L2 norm (1D over B*N)
 *   U2: batched_unfactored_clip_ema_apply — RMS clip, exp_avg EMA, param update (1D over B*N)
 *
 * Compilation: NVCC only. NO torch/ATen headers.
 * Interface: extern "C" launcher functions called from came_op.cpp (MSVC).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

// ============================================================================
// Helpers (copied from came_cuda_kernel.cu to keep both .cu files independent)
// ============================================================================

__device__ __forceinline__ int next_pow2(int x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

__device__ __forceinline__ float read_dtype(const void* ptr, int idx, int dtype) {
    if (dtype == 2)      return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
    else if (dtype == 1) return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
    else                 return reinterpret_cast<const float*>(ptr)[idx];
}

__device__ __forceinline__ void write_dtype(void* ptr, int idx, float val, int dtype) {
    if (dtype == 2)      reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16(val);
    else if (dtype == 1) reinterpret_cast<__half*>(ptr)[idx] = __float2half(val);
    else                 reinterpret_cast<float*>(ptr)[idx] = val;
}

__device__ __forceinline__ void block_reduce(float* sdata, int tid, int bdim) {
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
}

enum { DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_BF16 = 2 };

// ============================================================================
// P0: batched_prepare_param
// ============================================================================

__global__ void batched_prepare_param_kernel(
    const void* __restrict__ param_in,    // (B, R, C) any dtype
    float* __restrict__ param_fp32_out,   // (B, R, C) fp32
    float* scalar_buf,                     // (B, scalar_buf_size_per) fp32
    int total_N,
    int scalar_buf_size,
    float wd_lr, int dtype)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = i; idx < total_N; idx += stride) {
        float val = read_dtype(param_in, idx, dtype);
        if (wd_lr != 0.0f) val *= (1.0f - wd_lr);
        param_fp32_out[idx] = val;
    }

    if (i == 0) {
        for (int k = 0; k < scalar_buf_size; k++) {
            scalar_buf[k] = 0.0f;
        }
    }
}

// ============================================================================
// S0: batched_scale2
// ============================================================================

__global__ void batched_scale2_kernel(
    float* __restrict__ sq_col,    // (B, C)
    float* __restrict__ res_col,   // (B, C)
    int total_BC,
    float beta1, float beta2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_BC) {
        sq_col[i] *= beta1;
        res_col[i] *= beta2;
    }
}

// ============================================================================
// F1: batched_compute_sq_ema
// ============================================================================

__global__ void batched_compute_sq_ema_kernel(
    const float* __restrict__ grad,      // (B, R, C)
    float* __restrict__ sq_row,          // (B, R)
    float* __restrict__ sq_col,          // (B, C)
    float* __restrict__ scalar_buf,      // (B, 3)
    int B, int R, int C,
    float beta1, float one_minus_beta1, float eps0)
{
    int b = blockIdx.x;  // batch index
    int r = blockIdx.y;  // row index
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int RC = R * C;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    float contrib_scale = one_minus_beta1 / static_cast<float>(R);

    for (int c = tid; c < C; c += bdim) {
        float g = grad[b * RC + r * C + c];
        float val = g * g + eps0;
        local_sum += val;
        atomicAdd(&sq_col[b * C + c], val * contrib_scale);
    }

    sdata[tid] = local_sum;
    block_reduce(sdata, tid, bdim);

    if (tid == 0) {
        float row_mean = sdata[0] / static_cast<float>(C);
        sq_row[b * R + r] = beta1 * sq_row[b * R + r] + one_minus_beta1 * row_mean;
        atomicAdd(&scalar_buf[b * 3 + 0], sq_row[b * R + r]);
    }
}

// ============================================================================
// F2: batched_first_approx_and_norm
// ============================================================================

__global__ void batched_first_approx_and_norm_kernel(
    const float* __restrict__ grad,       // (B, R, C)
    const float* __restrict__ sq_row,     // (B, R)
    const float* __restrict__ sq_col,     // (B, C)
    float* __restrict__ approx_buf,       // (B, R, C)
    float* __restrict__ scalar_buf,       // (B, 3)
    int B, int R, int C)
{
    int b = blockIdx.x;
    int r = blockIdx.y;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int RC = R * C;

    extern __shared__ float sdata[];

    float row_mean = scalar_buf[b * 3 + 0] / static_cast<float>(R);
    float sq_row_r = sq_row[b * R + r];
    float ratio = (row_mean != 0.0f) ? (sq_row_r / row_mean) : 1.0f;
    float r_factor = rsqrtf(ratio);

    float local_l2 = 0.0f;

    for (int c = tid; c < C; c += bdim) {
        float c_factor = rsqrtf(sq_col[b * C + c]);
        float approx = r_factor * c_factor * grad[b * RC + r * C + c];
        approx_buf[b * RC + r * C + c] = approx;
        local_l2 += approx * approx;
    }

    sdata[tid] = local_l2;
    block_reduce(sdata, tid, bdim);

    if (tid == 0) {
        atomicAdd(&scalar_buf[b * 3 + 1], sdata[0]);
    }
}

// ============================================================================
// F3: batched_clip_ema_residual_ema
// ============================================================================

__global__ void batched_clip_ema_residual_ema_kernel(
    const float* __restrict__ approx_buf,   // (B, R, C)
    float* __restrict__ exp_avg,            // (B, R, C)
    float* __restrict__ res_row,            // (B, R)
    float* __restrict__ res_col,            // (B, C)
    float* __restrict__ scalar_buf,         // (B, 3)
    int B, int R, int C,
    float beta0, float one_minus_beta0,
    float beta2, float one_minus_beta2,
    float eps1, float clip_threshold)
{
    int b = blockIdx.x;
    int r = blockIdx.y;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int RC = R * C;

    extern __shared__ float sdata[];

    float l2_norm_sq = scalar_buf[b * 3 + 1];
    float inv_numel = 1.0f / static_cast<float>(R * C);
    float rms = sqrtf(l2_norm_sq * inv_numel);
    float inv_clamp = (rms > clip_threshold) ? (clip_threshold / rms) : 1.0f;

    float local_row_sum = 0.0f;
    float contrib_scale = one_minus_beta2 / static_cast<float>(R);

    for (int c = tid; c < C; c += bdim) {
        int idx = b * RC + r * C + c;
        float clipped = approx_buf[idx] * inv_clamp;

        float old_ea = exp_avg[idx];
        float new_ea = beta0 * old_ea + one_minus_beta0 * clipped;
        exp_avg[idx] = new_ea;

        float diff = clipped - new_ea;
        float res = diff * diff + eps1;
        local_row_sum += res;

        atomicAdd(&res_col[b * C + c], res * contrib_scale);
    }

    sdata[tid] = local_row_sum;
    block_reduce(sdata, tid, bdim);

    if (tid == 0) {
        float row_res_mean = sdata[0] / static_cast<float>(C);
        res_row[b * R + r] = beta2 * res_row[b * R + r] + one_minus_beta2 * row_res_mean;
        atomicAdd(&scalar_buf[b * 3 + 2], res_row[b * R + r]);
    }
}

// ============================================================================
// F4: batched_second_approx_and_apply
// ============================================================================

__global__ void batched_second_approx_and_apply_kernel(
    const float* __restrict__ res_row,      // (B, R)
    const float* __restrict__ res_col,      // (B, C)
    const float* __restrict__ exp_avg,      // (B, R, C)
    const float* __restrict__ param_fp32,   // (B, R, C)
    void* __restrict__ param_out,           // (B, R, C) any dtype
    const float* __restrict__ scalar_buf,   // (B, 3)
    int B, int R, int C,
    float lr, int dtype)
{
    int b = blockIdx.x;
    int r = blockIdx.y;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int RC = R * C;

    float res_row_mean = scalar_buf[b * 3 + 2] / static_cast<float>(R);
    float res_row_r = res_row[b * R + r];
    float ratio2 = (res_row_mean != 0.0f) ? (res_row_r / res_row_mean) : 1.0f;
    float r_factor2 = rsqrtf(ratio2);

    for (int c = tid; c < C; c += bdim) {
        int idx = b * RC + r * C + c;
        float c_factor2 = rsqrtf(res_col[b * C + c]);
        float update = r_factor2 * c_factor2 * exp_avg[idx];
        float result = param_fp32[idx] - update * lr;
        write_dtype(param_out, idx, result, dtype);
    }
}

// ============================================================================
// U1: batched_unfactored_sq_rsqrt_and_norm
// ============================================================================

__global__ void batched_unfactored_sq_rsqrt_and_norm_kernel(
    const float* __restrict__ grad,         // (B, N)
    float* __restrict__ exp_avg_sq,         // (B, N)
    float* __restrict__ buf,                // (B, N)
    float* __restrict__ scalar_buf,         // (B,)  — one scalar per batch item
    int B, int N,
    float beta1, float one_minus_beta1, float eps0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = B * N;

    for (int idx = i; idx < total; idx += stride) {
        int b = idx / N;
        float g = grad[idx];
        float val = g * g + eps0;
        exp_avg_sq[idx] = beta1 * exp_avg_sq[idx] + one_minus_beta1 * val;
        float rsqrt_val = rsqrtf(exp_avg_sq[idx]);
        float bv = rsqrt_val * g;
        buf[idx] = bv;
        atomicAdd(&scalar_buf[b], bv * bv);
    }
}

// ============================================================================
// U2: batched_unfactored_clip_ema_apply
// ============================================================================

__global__ void batched_unfactored_clip_ema_apply_kernel(
    const float* __restrict__ buf,          // (B, N)
    float* __restrict__ exp_avg,            // (B, N)
    const float* __restrict__ param_fp32,   // (B, N)
    void* __restrict__ param_out,           // (B, N) any dtype
    const float* __restrict__ scalar_buf,   // (B,)
    int B, int N,
    float beta0, float one_minus_beta0,
    float lr, float clip_threshold, int dtype)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = B * N;
    float inv_numel = 1.0f / static_cast<float>(N);

    for (int idx = i; idx < total; idx += stride) {
        int b = idx / N;
        float l2_norm_sq = scalar_buf[b];
        float rms = sqrtf(l2_norm_sq * inv_numel);
        float inv_clamp = (rms > clip_threshold) ? (clip_threshold / rms) : 1.0f;

        float clipped = buf[idx] * inv_clamp;
        float old_ea = exp_avg[idx];
        float new_ea = beta0 * old_ea + one_minus_beta0 * clipped;
        exp_avg[idx] = new_ea;
        float result = param_fp32[idx] - new_ea * lr;
        write_dtype(param_out, idx, result, dtype);
    }
}

// ============================================================================
// Launcher functions (extern "C" interface for came_op.cpp)
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
        }                                                                       \
    } while (0)

static int compute_block_size(int C) {
    int bs = 1;
    while (bs * 2 <= C && bs * 2 <= 1024) bs *= 2;
    return bs;
}

static int compute_grid_1d(int N, int block) {
    return (N + block - 1) / block;
}

extern "C" {

void launch_batched_prepare_param(
    const void* param_in,
    float* param_fp32_out,
    float* scalar_buf,
    int total_N, int scalar_buf_size,
    float wd_lr, int dtype,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(total_N, block);
    batched_prepare_param_kernel<<<grid, block, 0, stream>>>(
        param_in, param_fp32_out, scalar_buf, total_N, scalar_buf_size, wd_lr, dtype);
}

void launch_batched_scale2(
    float* sq_col, float* res_col,
    int total_BC,
    float beta1, float beta2,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(total_BC, block);
    batched_scale2_kernel<<<grid, block, 0, stream>>>(sq_col, res_col, total_BC, beta1, beta2);
}

void launch_batched_compute_sq_ema(
    const float* grad,
    float* sq_row, float* sq_col,
    float* scalar_buf,
    int B, int R, int C,
    float beta1, float one_minus_beta1, float eps0,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    const int smem = block * sizeof(float);
    dim3 grid(B, R);
    batched_compute_sq_ema_kernel<<<grid, block, smem, stream>>>(
        grad, sq_row, sq_col, scalar_buf, B, R, C, beta1, one_minus_beta1, eps0);
}

void launch_batched_first_approx_and_norm(
    const float* grad,
    const float* sq_row, const float* sq_col,
    float* approx_buf,
    float* scalar_buf,
    int B, int R, int C,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    const int smem = block * sizeof(float);
    dim3 grid(B, R);
    batched_first_approx_and_norm_kernel<<<grid, block, smem, stream>>>(
        grad, sq_row, sq_col, approx_buf, scalar_buf, B, R, C);
}

void launch_batched_clip_ema_residual_ema(
    const float* approx_buf,
    float* exp_avg,
    float* res_row, float* res_col,
    float* scalar_buf,
    int B, int R, int C,
    float beta0, float one_minus_beta0,
    float beta2, float one_minus_beta2,
    float eps1, float clip_threshold,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    const int smem = block * sizeof(float);
    dim3 grid(B, R);
    batched_clip_ema_residual_ema_kernel<<<grid, block, smem, stream>>>(
        approx_buf, exp_avg, res_row, res_col, scalar_buf,
        B, R, C, beta0, one_minus_beta0, beta2, one_minus_beta2, eps1, clip_threshold);
}

void launch_batched_second_approx_and_apply(
    const float* res_row, const float* res_col,
    const float* exp_avg,
    const float* param_fp32,
    void* param_out,
    const float* scalar_buf,
    int B, int R, int C,
    float lr, int dtype,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    dim3 grid(B, R);
    batched_second_approx_and_apply_kernel<<<grid, block, 0, stream>>>(
        res_row, res_col, exp_avg, param_fp32, param_out, scalar_buf,
        B, R, C, lr, dtype);
}

void launch_batched_unfactored_sq_rsqrt_and_norm(
    const float* grad,
    float* exp_avg_sq,
    float* buf,
    float* scalar_buf,
    int B, int N,
    float beta1, float one_minus_beta1, float eps0,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(B * N, block);
    batched_unfactored_sq_rsqrt_and_norm_kernel<<<grid, block, 0, stream>>>(
        grad, exp_avg_sq, buf, scalar_buf, B, N, beta1, one_minus_beta1, eps0);
}

void launch_batched_unfactored_clip_ema_apply(
    const float* buf,
    float* exp_avg,
    const float* param_fp32,
    void* param_out,
    const float* scalar_buf,
    int B, int N,
    float beta0, float one_minus_beta0,
    float lr, float clip_threshold, int dtype,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(B * N, block);
    batched_unfactored_clip_ema_apply_kernel<<<grid, block, 0, stream>>>(
        buf, exp_avg, param_fp32, param_out, scalar_buf,
        B, N, beta0, one_minus_beta0, lr, clip_threshold, dtype);
}

} // extern "C"
