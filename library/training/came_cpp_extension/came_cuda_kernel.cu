/**
 * came_cuda_kernel.cu — Fused CUDA kernels for CAME optimizer.
 *
 * 8 kernels total:
 *   P0: prepare_param        — dtype conversion + weight decay + scalar_buf zeroing
 *   S0: scale2               — column state pre-multiply + scalar_buf zeroing (factored only)
 *   F1: compute_sq_ema       — grad^2+eps0, row/col EMA for exp_avg_sq
 *   F2: first_approx_and_norm — first approx_sq_grad * grad, L2 norm accumulation
 *   F3: clip_ema_residual_ema — RMS clip, exp_avg EMA, residual row/col EMA
 *   F4: second_approx_and_apply — second approx_sq_grad * exp_avg, param update + dtype writeback
 *   U1: unfactored_sq_rsqrt  — exp_avg_sq EMA, rsqrt*grad, L2 norm
 *   U2: unfactored_clip_apply — RMS clip, exp_avg EMA, param update + dtype writeback
 *
 * Compilation: NVCC only. NO torch/ATen headers.
 * Interface: extern "C" launcher functions called from came_op.cpp (MSVC).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

// ============================================================================
// Helpers
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

// Block-level reduction in shared memory. After call, sdata[0] = sum.
__device__ __forceinline__ void block_reduce(float* sdata, int tid, int bdim) {
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
}

// Constants for dtype_flag
enum { DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_BF16 = 2 };

// ============================================================================
// P0: prepare_param
// ============================================================================

__global__ void prepare_param_kernel(
    const void* __restrict__ param_in,
    float* __restrict__ param_fp32_out,
    float* scalar_buf,
    int N, int scalar_buf_size,
    float wd_lr, int dtype)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = i; idx < N; idx += stride) {
        float val = read_dtype(param_in, idx, dtype);
        if (wd_lr != 0.0f) val *= (1.0f - wd_lr);
        param_fp32_out[idx] = val;
    }

    // Last thread clears scalar_buf
    if (i == 0) {
        for (int k = 0; k < scalar_buf_size; k++) {
            scalar_buf[k] = 0.0f;
        }
    }
}

// ============================================================================
// S0: scale2
// ============================================================================

__global__ void scale2_kernel(
    float* __restrict__ sq_col,
    float* __restrict__ res_col,
    int C,
    float beta1, float beta2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C) {
        sq_col[i] *= beta1;
        res_col[i] *= beta2;
    }
}

// ============================================================================
// F1: compute_sq_ema
// ============================================================================

__global__ void compute_sq_ema_kernel(
    const float* __restrict__ grad,
    float* __restrict__ sq_row,
    float* __restrict__ sq_col,
    float* __restrict__ scalar_buf,
    int R, int C,
    float beta1, float one_minus_beta1, float eps0)
{
    // One block per row
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    float contrib_scale = one_minus_beta1 / static_cast<float>(R);

    for (int c = tid; c < C; c += bdim) {
        float g = grad[r * C + c];
        float val = g * g + eps0;
        local_sum += val;
        // Column EMA contribution: atomicAdd (1-beta1)/R * val
        atomicAdd(&sq_col[c], val * contrib_scale);
    }

    sdata[tid] = local_sum;
    block_reduce(sdata, tid, bdim);

    // Thread 0 writes row EMA
    if (tid == 0) {
        float row_mean = sdata[0] / static_cast<float>(C);
        sq_row[r] = beta1 * sq_row[r] + one_minus_beta1 * row_mean;
        atomicAdd(&scalar_buf[0], sq_row[r]);
    }
}

// ============================================================================
// F2: first_approx_and_norm
// ============================================================================

__global__ void first_approx_and_norm_kernel(
    const float* __restrict__ grad,
    const float* __restrict__ sq_row,
    const float* __restrict__ sq_col,
    float* __restrict__ approx_buf,
    float* __restrict__ scalar_buf,
    int R, int C)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float sdata[];

    float row_mean = scalar_buf[0] / static_cast<float>(R);
    float sq_row_r = sq_row[r];
    float ratio = (row_mean != 0.0f) ? (sq_row_r / row_mean) : 1.0f;
    float r_factor = rsqrtf(ratio);

    float local_l2 = 0.0f;

    for (int c = tid; c < C; c += bdim) {
        float c_factor = rsqrtf(sq_col[c]);
        float approx = r_factor * c_factor * grad[r * C + c];
        approx_buf[r * C + c] = approx;
        local_l2 += approx * approx;
    }

    sdata[tid] = local_l2;
    block_reduce(sdata, tid, bdim);

    if (tid == 0) {
        atomicAdd(&scalar_buf[1], sdata[0]);
    }
}

// ============================================================================
// F3: clip_ema_residual_ema
// ============================================================================

__global__ void clip_ema_residual_ema_kernel(
    const float* __restrict__ approx_buf,
    float* __restrict__ exp_avg,
    float* __restrict__ res_row,
    float* __restrict__ res_col,
    float* __restrict__ scalar_buf,
    int R, int C,
    float beta0, float one_minus_beta0,
    float beta2, float one_minus_beta2,
    float eps1, float clip_threshold)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float sdata[];

    float l2_norm_sq = scalar_buf[1];
    float inv_numel = 1.0f / static_cast<float>(R * C);
    float rms = sqrtf(l2_norm_sq * inv_numel);
    float inv_clamp = (rms > clip_threshold) ? (clip_threshold / rms) : 1.0f;

    float local_row_sum = 0.0f;
    float contrib_scale = one_minus_beta2 / static_cast<float>(R);

    for (int c = tid; c < C; c += bdim) {
        int idx = r * C + c;
        float clipped = approx_buf[idx] * inv_clamp;

        // exp_avg EMA
        float old_ea = exp_avg[idx];
        float new_ea = beta0 * old_ea + one_minus_beta0 * clipped;
        exp_avg[idx] = new_ea;

        // Residual
        float diff = clipped - new_ea;
        float res = diff * diff + eps1;
        local_row_sum += res;

        // Column residual EMA contribution
        atomicAdd(&res_col[c], res * contrib_scale);
    }

    sdata[tid] = local_row_sum;
    block_reduce(sdata, tid, bdim);

    if (tid == 0) {
        float row_res_mean = sdata[0] / static_cast<float>(C);
        res_row[r] = beta2 * res_row[r] + one_minus_beta2 * row_res_mean;
        atomicAdd(&scalar_buf[2], res_row[r]);
    }
}

// ============================================================================
// F4: second_approx_and_apply
// ============================================================================

__global__ void second_approx_and_apply_kernel(
    const float* __restrict__ res_row,
    const float* __restrict__ res_col,
    const float* __restrict__ exp_avg,
    const float* __restrict__ param_fp32,
    void* __restrict__ param_out,
    const float* __restrict__ scalar_buf,
    int R, int C,
    float lr, int dtype)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    float res_row_mean = scalar_buf[2] / static_cast<float>(R);
    float res_row_r = res_row[r];
    float ratio2 = (res_row_mean != 0.0f) ? (res_row_r / res_row_mean) : 1.0f;
    float r_factor2 = rsqrtf(ratio2);

    for (int c = tid; c < C; c += bdim) {
        int idx = r * C + c;
        float c_factor2 = rsqrtf(res_col[c]);
        float update = r_factor2 * c_factor2 * exp_avg[idx];
        float result = param_fp32[idx] - update * lr;
        write_dtype(param_out, idx, result, dtype);
    }
}

// ============================================================================
// U1: unfactored_sq_rsqrt_and_norm
// ============================================================================

__global__ void unfactored_sq_rsqrt_and_norm_kernel(
    const float* __restrict__ grad,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ buf,
    float* __restrict__ scalar_buf,
    int N,
    float beta1, float one_minus_beta1, float eps0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = i; idx < N; idx += stride) {
        float g = grad[idx];
        float val = g * g + eps0;
        exp_avg_sq[idx] = beta1 * exp_avg_sq[idx] + one_minus_beta1 * val;
        float rsqrt_val = rsqrtf(exp_avg_sq[idx]);
        float b = rsqrt_val * g;
        buf[idx] = b;
        atomicAdd(&scalar_buf[0], b * b);
    }
}

// ============================================================================
// U2: unfactored_clip_ema_apply
// ============================================================================

__global__ void unfactored_clip_ema_apply_kernel(
    const float* __restrict__ buf,
    float* __restrict__ exp_avg,
    const float* __restrict__ param_fp32,
    void* __restrict__ param_out,
    const float* __restrict__ scalar_buf,
    int N,
    float beta0, float one_minus_beta0,
    float lr, float clip_threshold, int dtype)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float l2_norm_sq = scalar_buf[0];
    float inv_numel = 1.0f / static_cast<float>(N);
    float rms = sqrtf(l2_norm_sq * inv_numel);
    float inv_clamp = (rms > clip_threshold) ? (clip_threshold / rms) : 1.0f;

    for (int idx = i; idx < N; idx += stride) {
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

void launch_prepare_param(
    const void* param_in,
    float* param_fp32_out,
    float* scalar_buf,
    int N, int scalar_buf_size,
    float wd_lr, int dtype,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(N, block);
    prepare_param_kernel<<<grid, block, 0, stream>>>(
        param_in, param_fp32_out, scalar_buf, N, scalar_buf_size, wd_lr, dtype);
}

void launch_scale2(
    float* sq_col, float* res_col,
    int C,
    float beta1, float beta2,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(C, block);
    scale2_kernel<<<grid, block, 0, stream>>>(sq_col, res_col, C, beta1, beta2);
}

void launch_compute_sq_ema(
    const float* grad,
    float* sq_row, float* sq_col,
    float* scalar_buf,
    int R, int C,
    float beta1, float one_minus_beta1, float eps0,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    const int smem = block * sizeof(float);
    compute_sq_ema_kernel<<<R, block, smem, stream>>>(
        grad, sq_row, sq_col, scalar_buf, R, C, beta1, one_minus_beta1, eps0);
}

void launch_first_approx_and_norm(
    const float* grad,
    const float* sq_row, const float* sq_col,
    float* approx_buf,
    float* scalar_buf,
    int R, int C,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    const int smem = block * sizeof(float);
    first_approx_and_norm_kernel<<<R, block, smem, stream>>>(
        grad, sq_row, sq_col, approx_buf, scalar_buf, R, C);
}

void launch_clip_ema_residual_ema(
    const float* approx_buf,
    float* exp_avg,
    float* res_row, float* res_col,
    float* scalar_buf,
    int R, int C,
    float beta0, float one_minus_beta0,
    float beta2, float one_minus_beta2,
    float eps1, float clip_threshold,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    const int smem = block * sizeof(float);
    clip_ema_residual_ema_kernel<<<R, block, smem, stream>>>(
        approx_buf, exp_avg, res_row, res_col, scalar_buf,
        R, C, beta0, one_minus_beta0, beta2, one_minus_beta2, eps1, clip_threshold);
}

void launch_second_approx_and_apply(
    const float* res_row, const float* res_col,
    const float* exp_avg,
    const float* param_fp32,
    void* param_out,
    const float* scalar_buf,
    int R, int C,
    float lr, int dtype,
    cudaStream_t stream)
{
    const int block = compute_block_size(C);
    second_approx_and_apply_kernel<<<R, block, 0, stream>>>(
        res_row, res_col, exp_avg, param_fp32, param_out, scalar_buf,
        R, C, lr, dtype);
}

void launch_unfactored_sq_rsqrt_and_norm(
    const float* grad,
    float* exp_avg_sq,
    float* buf,
    float* scalar_buf,
    int N,
    float beta1, float one_minus_beta1, float eps0,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(N, block);
    unfactored_sq_rsqrt_and_norm_kernel<<<grid, block, 0, stream>>>(
        grad, exp_avg_sq, buf, scalar_buf, N, beta1, one_minus_beta1, eps0);
}

void launch_unfactored_clip_ema_apply(
    const float* buf,
    float* exp_avg,
    const float* param_fp32,
    void* param_out,
    const float* scalar_buf,
    int N,
    float beta0, float one_minus_beta0,
    float lr, float clip_threshold, int dtype,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = compute_grid_1d(N, block);
    unfactored_clip_ema_apply_kernel<<<grid, block, 0, stream>>>(
        buf, exp_avg, param_fp32, param_out, scalar_buf,
        N, beta0, one_minus_beta0, lr, clip_threshold, dtype);
}

} // extern "C"
