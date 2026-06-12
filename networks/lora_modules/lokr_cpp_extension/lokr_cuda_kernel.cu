/**
 * lokr_cuda_kernel.cu — Fused CUDA kernels for LoKR (Kronecker-factored LoRA).
 *
 * 4 kernels total:
 *   kron1_fwd — Single-stage fused forward  (2 GEMMs + cast + scalar → 1 launch)
 *   kron1_bwd — Single-stage fused backward (temp recompute + 4 grad GEMMs → 1 launch)
 *   kron2_fwd — Two-stage fused forward     (4 GEMMs + cast + scalar → 1 launch)
 *   kron2_bwd — Two-stage fused backward    (2 recompute + 8 grad GEMMs → 1 launch)
 *
 * Compilation: NVCC only. NO torch/ATen headers.
 * Interface: extern "C" launcher functions called from lokr_op.cpp (MSVC).
 *
 * Design:
 *   - One block per batch element (B = leading dims product)
 *   - Threads handle k (output columns) values within each block
 *   - Weight matrices in shared memory (tiny); x streamed from L1 cache
 *   - temp (intermediate GEMM result) in shared memory for stage-to-stage register bypass
 *   - All internal accumulation in fp32; dtype conversion at read/write boundaries
 *   - Backward: grad_w1 via atomicAdd (small target set), grad_w2 via materialized
 *     grad_temp (computed in kernel, reduced by ATen matmul in lokr_op.cpp)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Helpers (copied from came_cuda_kernel.cu — identical interface)
// ============================================================================

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

// Constants for dtype_flag
enum { DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_BF16 = 2 };

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
        }                                                                       \
    } while (0)

// ============================================================================
// kron1_fwd_kernel — Single-stage fused forward
//
//   temp[r, k] = sum_s X[r, s] * w2[k, s]           (GEMM, K=in_n)
//   out[p, k]  = scalar * sum_r w1[p, r] * temp[r,k] (GEMM, K=in_m)
//
// Shared memory: w1[out_l*in_m] + temp[in_m*out_k]
// ============================================================================

__global__ void kron1_fwd_kernel(
    const void* __restrict__ x_ptr,    // (B, in_m*in_n) — any dtype
    const void* __restrict__ w1_ptr,   // (out_l, in_m)  — any dtype
    const void* __restrict__ w2_ptr,   // (out_k, in_n)  — any dtype
    void* __restrict__ out_ptr,        // (B, out_l*out_k) — x dtype
    int B, int in_m, int in_n, int out_l, int out_k,
    float scalar,
    int x_dtype, int w1_dtype, int w2_dtype)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];
    float* w1_sm   = smem;                       // out_l * in_m
    float* temp_sm = w1_sm + out_l * in_m;       // in_m  * out_k

    // Phase 0: Load w1 to shared memory
    int w1_size = out_l * in_m;
    for (int i = tid; i < w1_size; i += bdim)
        w1_sm[i] = read_dtype(w1_ptr, i, w1_dtype);

    // Phase 1: temp[r,k] = sum_s x[b,r,s] * w2[k,s]
    // Each thread handles k = tid, tid+bdim, ...
    for (int k = tid; k < out_k; k += bdim) {
        for (int r = 0; r < in_m; r++) {
            float sum = 0.0f;
            int x_base = b * (in_m * in_n) + r * in_n;
            int w2_base = k * in_n;
            for (int s = 0; s < in_n; s++)
                sum += read_dtype(x_ptr, x_base + s, x_dtype) *
                       read_dtype(w2_ptr, w2_base + s, w2_dtype);
            temp_sm[r * out_k + k] = sum;
        }
    }
    __syncthreads();

    // Phase 2: out[b,p,k] = scalar * sum_r w1[p,r] * temp[r,k]
    for (int k = tid; k < out_k; k += bdim) {
        for (int p = 0; p < out_l; p++) {
            float sum = 0.0f;
            for (int r = 0; r < in_m; r++)
                sum += w1_sm[p * in_m + r] * temp_sm[r * out_k + k];
            write_dtype(out_ptr, b * (out_l * out_k) + p * out_k + k,
                        scalar * sum, x_dtype);
        }
    }
}

// ============================================================================
// kron1_bwd_kernel — Single-stage fused backward
//
// Recomputes temp, then computes grad_x, grad_w1 (atomicAdd), grad_scalar
// (atomicAdd), and materializes grad_temp for external grad_w2 computation.
//
// Shared memory: w1[out_l*in_m] + temp[in_m*out_k] (reused as grad_temp)
// ============================================================================

__global__ void kron1_bwd_kernel(
    const void* __restrict__ grad_out_ptr,  // (B, out_l*out_k) — any dtype
    const void* __restrict__ x_ptr,          // (B, in_m*in_n)  — any dtype
    const void* __restrict__ w1_ptr,         // (out_l, in_m)   — any dtype
    const void* __restrict__ w2_ptr,         // (out_k, in_n)   — any dtype
    void* __restrict__ grad_x_ptr,           // (B, in_m*in_n)  — x dtype
    float* __restrict__ grad_w1_ptr,         // (out_l, in_m)   — fp32, pre-zeroed
    float* __restrict__ grad_scalar_ptr,     // (1,)            — fp32, pre-zeroed
    float* __restrict__ grad_temp_ptr,       // (B, in_m, out_k)— fp32 workspace
    int B, int in_m, int in_n, int out_l, int out_k,
    float scalar,
    int grad_out_dtype, int x_dtype, int w1_dtype, int w2_dtype)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];
    float* w1_sm   = smem;                       // out_l * in_m
    float* temp_sm = w1_sm + out_l * in_m;       // in_m  * out_k (reused)

    // Phase 0: Load w1 to shared memory
    int w1_size = out_l * in_m;
    for (int i = tid; i < w1_size; i += bdim)
        w1_sm[i] = read_dtype(w1_ptr, i, w1_dtype);

    // Phase 1: Recompute temp[r,k] = sum_s x[b,r,s] * w2[k,s]
    for (int k = tid; k < out_k; k += bdim) {
        for (int r = 0; r < in_m; r++) {
            float sum = 0.0f;
            int x_base = b * (in_m * in_n) + r * in_n;
            int w2_base = k * in_n;
            for (int s = 0; s < in_n; s++)
                sum += read_dtype(x_ptr, x_base + s, x_dtype) *
                       read_dtype(w2_ptr, w2_base + s, w2_dtype);
            temp_sm[r * out_k + k] = sum;
        }
    }
    __syncthreads();

    // Phase 2a: grad_w1[p,r] += scalar * sum_k grad_out[b,p,k] * temp[r,k]
    // Threads handle (p,r) pairs
    int w1_elems = out_l * in_m;
    for (int idx = tid; idx < w1_elems; idx += bdim) {
        int p = idx / in_m;
        int r = idx % in_m;
        float sum = 0.0f;
        int go_base = b * (out_l * out_k) + p * out_k;
        for (int k = 0; k < out_k; k++)
            sum += read_dtype(grad_out_ptr, go_base + k, grad_out_dtype) *
                   temp_sm[r * out_k + k];
        atomicAdd(&grad_w1_ptr[p * in_m + r], scalar * sum);
    }

    // Phase 2b: grad_scalar += sum_{p,k} grad_out[b,p,k] * fwd_out_raw[p,k]
    //   fwd_out_raw[p,k] = sum_r w1[p,r] * temp[r,k]  (before scalar)
    {
        float local_gs = 0.0f;
        for (int k = tid; k < out_k; k += bdim) {
            for (int p = 0; p < out_l; p++) {
                float go = read_dtype(grad_out_ptr,
                    b * (out_l * out_k) + p * out_k + k, grad_out_dtype);
                float fwd_raw = 0.0f;
                for (int r = 0; r < in_m; r++)
                    fwd_raw += w1_sm[p * in_m + r] * temp_sm[r * out_k + k];
                local_gs += go * fwd_raw;
            }
        }
        atomicAdd(grad_scalar_ptr, local_gs);
    }

    __syncthreads();  // Done reading temp_sm; will overwrite with grad_temp

    // Phase 3: grad_temp[r,k] = scalar * sum_p w1[p,r] * grad_out[b,p,k]
    // Write to shared (overwrite temp) AND global (for external grad_w2)
    for (int k = tid; k < out_k; k += bdim) {
        for (int r = 0; r < in_m; r++) {
            float sum = 0.0f;
            for (int p = 0; p < out_l; p++)
                sum += w1_sm[p * in_m + r] *
                       read_dtype(grad_out_ptr,
                           b * (out_l * out_k) + p * out_k + k, grad_out_dtype);
            float val = scalar * sum;
            temp_sm[r * out_k + k] = val;
            grad_temp_ptr[b * (in_m * out_k) + r * out_k + k] = val;
        }
    }
    __syncthreads();

    // Phase 4: grad_x[b,r,s] = sum_k grad_temp[r,k] * w2[k,s]
    for (int s = tid; s < in_n; s += bdim) {
        for (int r = 0; r < in_m; r++) {
            float sum = 0.0f;
            for (int k = 0; k < out_k; k++)
                sum += temp_sm[r * out_k + k] *
                       read_dtype(w2_ptr, k * in_n + s, w2_dtype);
            write_dtype(grad_x_ptr, b * (in_m * in_n) + r * in_n + s,
                        sum, x_dtype);
        }
    }
}

// ============================================================================
// kron2_fwd_kernel — Two-stage fused forward (decompose_both)
//
//   T1[r,j]   = sum_s X[r,s] * w2_b[j,s]       (GEMM, K=in_n)
//   Z[i,j]    = sum_r w1_b[i,r] * T1[r,j]       (GEMM, K=in_m)
//   T2[i,k]   = sum_j Z[i,j] * w2_a[k,j]        (GEMM, K=r2)  [inline, no storage]
//   out[p,k]  = scalar * sum_i w1_a[p,i] * T2[i,k] (GEMM, K=r1)
//
// Saves temp1 (= T1) to global memory for backward.
//
// Shared memory: w1_a[r1*out_l] + w1_b[in_m*r1] + w2_b[in_n*r2] + T1/Z[max(in_m,r1)*r2]
// w2_a streamed from global (L1 cached).
// ============================================================================

__global__ void kron2_fwd_kernel(
    const void* __restrict__ x_ptr,      // (B, in_m*in_n)
    const void* __restrict__ w1_a_ptr,   // (out_l, r1)
    const void* __restrict__ w1_b_ptr,   // (r1, in_m)
    const void* __restrict__ w2_a_ptr,   // (out_k, r2)
    const void* __restrict__ w2_b_ptr,   // (r2, in_n)
    void* __restrict__ out_ptr,          // (B, out_l*out_k)
    float* __restrict__ temp1_ptr,       // (B, in_m, r2) — fp32, saved for backward
    int B, int in_m, int in_n, int out_l, int out_k, int r1, int r2,
    float scalar,
    int x_dtype, int w1a_dtype, int w1b_dtype, int w2a_dtype, int w2b_dtype)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];
    float* w1a_sm = smem;                              // out_l * r1
    float* w1b_sm = w1a_sm + out_l * r1;               // r1 * in_m
    float* w2b_sm = w1b_sm + r1 * in_m;                // r2 * in_n
    // T1 and Z share space (T1 freed before Z needed)
    int tz_size = in_m * r2 > r1 * r2 ? in_m * r2 : r1 * r2;
    float* tz_sm = w2b_sm + r2 * in_n;                 // max(in_m, r1) * r2

    // Phase 0: Load weight matrices to shared memory
    int w1a_size = out_l * r1;
    for (int i = tid; i < w1a_size; i += bdim)
        w1a_sm[i] = read_dtype(w1_a_ptr, i, w1a_dtype);

    int w1b_size = r1 * in_m;
    for (int i = tid; i < w1b_size; i += bdim)
        w1b_sm[i] = read_dtype(w1_b_ptr, i, w1b_dtype);

    int w2b_size = r2 * in_n;
    for (int i = tid; i < w2b_size; i += bdim)
        w2b_sm[i] = read_dtype(w2_b_ptr, i, w2b_dtype);

    // Phase 1: T1[r,j] = sum_s x[b,r,s] * w2_b[j,s]  → tz_sm (as T1)
    for (int j = tid; j < r2; j += bdim) {
        for (int r = 0; r < in_m; r++) {
            float sum = 0.0f;
            int x_base = b * (in_m * in_n) + r * in_n;
            for (int s = 0; s < in_n; s++)
                sum += read_dtype(x_ptr, x_base + s, x_dtype) *
                       w2b_sm[j * in_n + s];
            tz_sm[r * r2 + j] = sum;
            // Also write temp1 to global for backward
            temp1_ptr[b * (in_m * r2) + r * r2 + j] = sum;
        }
    }
    __syncthreads();

    // Phase 2: Z[i,j] = sum_r w1_b[i,r] * T1[r,j] → overwrite tz_sm start
    // T1 (in_m*r2 entries) is no longer needed in shared after this;
    // it's saved in temp1_ptr global. Z uses r1*r2 entries.
    for (int idx = tid; idx < r1 * r2; idx += bdim) {
        int i = idx / r2;
        int j = idx % r2;
        float sum = 0.0f;
        for (int r = 0; r < in_m; r++)
            sum += w1b_sm[i * in_m + r] * tz_sm[r * r2 + j];
        tz_sm[i * r2 + j] = sum;  // overwrite T1 region
    }
    __syncthreads();

    // Phase 3+4: out[p,k] = scalar * sum_i w1_a[p,i] * temp2[i,k]
    //   where temp2[i,k] = sum_j Z[i,j] * w2_a[k,j]
    // Z is precomputed in tz_sm — eliminates 220x redundant FLOPs
    for (int k = tid; k < out_k; k += bdim) {
        // Compute temp2[i,k] for all i into registers
        float temp2[16];  // r1 <= 16 in practice
        for (int i = 0; i < r1; i++) {
            float sum = 0.0f;
            for (int j = 0; j < r2; j++)
                sum += tz_sm[i * r2 + j] *
                       read_dtype(w2_a_ptr, k * r2 + j, w2a_dtype);
            temp2[i] = sum;
        }
        for (int p = 0; p < out_l; p++) {
            float result = 0.0f;
            for (int i = 0; i < r1; i++)
                result += w1a_sm[p * r1 + i] * temp2[i];
            write_dtype(out_ptr, b * (out_l * out_k) + p * out_k + k,
                        scalar * result, x_dtype);
        }
    }
}

// ============================================================================
// kron2_bwd_kernel — Two-stage fused backward (decompose_both)
//
// Uses saved temp1 to recompute Z and temp2, then computes all gradients.
//
// Shared memory: w1_a + w1_b + w2_a + w2_b + workspace
// ============================================================================

__global__ void kron2_bwd_kernel(
    const void* __restrict__ grad_out_ptr,   // (B, out_l*out_k)
    const float* __restrict__ temp1_ptr,      // (B, in_m, r2) — saved from forward
    const void* __restrict__ x_ptr,           // (B, in_m*in_n)
    const void* __restrict__ w1_a_ptr,        // (out_l, r1)
    const void* __restrict__ w1_b_ptr,        // (r1, in_m)
    const void* __restrict__ w2_a_ptr,        // (out_k, r2)
    const void* __restrict__ w2_b_ptr,        // (r2, in_n)
    void* __restrict__ grad_x_ptr,            // (B, in_m*in_n)
    float* __restrict__ grad_w1a_ptr,         // (out_l, r1)  — fp32, pre-zeroed
    float* __restrict__ grad_w1b_ptr,         // (r1, in_m)   — fp32, pre-zeroed
    float* __restrict__ grad_w2a_ptr,         // (out_k, r2)  — fp32, pre-zeroed
    float* __restrict__ grad_w2b_ptr,         // (r2, in_n)   — fp32, pre-zeroed
    float* __restrict__ grad_scalar_ptr,      // (1,)          — fp32, pre-zeroed
    float* __restrict__ gt1_ptr,              // (B, in_m, r2) — fp32 workspace
    float* __restrict__ gt2_ptr,              // (B, r1, out_k)— fp32 workspace
    int B, int in_m, int in_n, int out_l, int out_k, int r1, int r2,
    float scalar,
    int grad_out_dtype, int x_dtype,
    int w1a_dtype, int w1b_dtype, int w2a_dtype, int w2b_dtype)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];
    float* w1a_sm   = smem;                                    // out_l * r1
    float* w1b_sm   = w1a_sm + out_l * r1;                     // r1 * in_m
    float* w2b_sm   = w1b_sm + r1 * in_m;                      // r2 * in_n
    float* temp1_sm = w2b_sm + r2 * in_n;                      // in_m * r2
    float* Z_sm     = temp1_sm + in_m * r2;                    // r1 * r2
    float* gradZ_sm = Z_sm + r1 * r2;                          // r1 * r2

    // Phase 0: Load weights and temp1 to shared memory
    for (int i = tid; i < out_l * r1; i += bdim)
        w1a_sm[i] = read_dtype(w1_a_ptr, i, w1a_dtype);
    for (int i = tid; i < r1 * in_m; i += bdim)
        w1b_sm[i] = read_dtype(w1_b_ptr, i, w1b_dtype);
    for (int i = tid; i < r2 * in_n; i += bdim)
        w2b_sm[i] = read_dtype(w2_b_ptr, i, w2b_dtype);
    for (int i = tid; i < in_m * r2; i += bdim)
        temp1_sm[i] = temp1_ptr[b * (in_m * r2) + i];
    __syncthreads();

    // Phase 0b: Precompute Z[i,j] = sum_m w1_b[i,m] * temp1[b,m,j] → Z_sm
    // This eliminates redundant Z recomputation for each k in Phase 1
    // (old code recomputed Z inside the k loop — 25x more FLOPs)
    for (int idx = tid; idx < r1 * r2; idx += bdim) {
        int i = idx / r2;
        int j = idx % r2;
        float sum = 0.0f;
        for (int m = 0; m < in_m; m++)
            sum += w1b_sm[i * in_m + m] * temp1_sm[m * r2 + j];
        Z_sm[i * r2 + j] = sum;
    }
    __syncthreads();

    // Phase 1: temp2[i,k] = sum_j Z[i,j] * w2_a[k,j] → gt2_ptr (global)
    // Simplified from 4-deep nested loop to 3-deep using precomputed Z_sm
    for (int k = tid; k < out_k; k += bdim) {
        for (int i = 0; i < r1; i++) {
            float sum = 0.0f;
            for (int j = 0; j < r2; j++)
                sum += Z_sm[i * r2 + j] *
                       read_dtype(w2_a_ptr, k * r2 + j, w2a_dtype);
            gt2_ptr[b * (r1 * out_k) + i * out_k + k] = sum;
        }
    }
    __syncthreads();  // Ensure gt2 writes are visible

    // Phase 2: grad_w1_a[p,i] += scalar * sum_k grad_out[b,p,k] * temp2[i,k]
    //          grad_scalar += sum_{p,k} grad_out[b,p,k] * fwd_out_raw[p,k]
    //            where fwd_out_raw[p,k] = sum_i w1_a[p,i] * temp2[i,k]
    for (int idx = tid; idx < out_l * r1; idx += bdim) {
        int p = idx / r1;
        int i = idx % r1;
        float sum = 0.0f;
        for (int k = 0; k < out_k; k++)
            sum += read_dtype(grad_out_ptr,
                b * (out_l * out_k) + p * out_k + k, grad_out_dtype) *
                gt2_ptr[b * (r1 * out_k) + i * out_k + k];
        atomicAdd(&grad_w1a_ptr[p * r1 + i], scalar * sum);
    }
    {
        float local_gs = 0.0f;
        for (int k = tid; k < out_k; k += bdim) {
            for (int p = 0; p < out_l; p++) {
                float go = read_dtype(grad_out_ptr,
                    b * (out_l * out_k) + p * out_k + k, grad_out_dtype);
                float fwd_raw = 0.0f;
                for (int i = 0; i < r1; i++)
                    fwd_raw += w1a_sm[p * r1 + i] *
                               gt2_ptr[b * (r1 * out_k) + i * out_k + k];
                local_gs += go * fwd_raw;
            }
        }
        atomicAdd(grad_scalar_ptr, local_gs);
    }
    __syncthreads();  // All threads done reading gt2 as temp2

    // Phase 3: grad_temp2[i,k] = scalar * sum_p w1_a[p,i] * grad_out[b,p,k]
    //   → gt2_ptr (overwrite temp2 with grad_temp2)
    for (int k = tid; k < out_k; k += bdim) {
        for (int i = 0; i < r1; i++) {
            float sum = 0.0f;
            for (int p = 0; p < out_l; p++)
                sum += w1a_sm[p * r1 + i] *
                       read_dtype(grad_out_ptr,
                           b * (out_l * out_k) + p * out_k + k, grad_out_dtype);
            gt2_ptr[b * (r1 * out_k) + i * out_k + k] = scalar * sum;
        }
    }
    __syncthreads();  // Ensure grad_temp2 writes are visible

    // Phase 4: Precompute grad_Z[i,j] = sum_k grad_temp2[i,k] * w2_a[k,j] → shared
    for (int idx = tid; idx < r1 * r2; idx += bdim) {
        int i = idx / r2;
        int j = idx % r2;
        float sum = 0.0f;
        for (int k = 0; k < out_k; k++)
            sum += gt2_ptr[b * (r1 * out_k) + i * out_k + k] *
                   read_dtype(w2_a_ptr, k * r2 + j, w2a_dtype);
        gradZ_sm[i * r2 + j] = sum;
    }
    __syncthreads();  // Ensure grad_Z is visible in shared

    // Phase 5: grad_w1_b[i,r] += sum_j grad_Z[i,j] * temp1[r,j]
    //          grad_temp1[r,j] = sum_i w1_b[i,r] * grad_Z[i,j] → gt1_ptr (global)
    // Note: grad_Z already includes scalar through the chain
    // (grad_temp2 → grad_Z), so NO extra scalar multiplication here
    for (int idx = tid; idx < r1 * in_m; idx += bdim) {
        int i = idx / in_m;
        int r = idx % in_m;
        float sum = 0.0f;
        for (int j = 0; j < r2; j++)
            sum += gradZ_sm[i * r2 + j] * temp1_sm[r * r2 + j];
        atomicAdd(&grad_w1b_ptr[i * in_m + r], sum);
    }
    for (int idx = tid; idx < in_m * r2; idx += bdim) {
        int r = idx / r2;
        int j = idx % r2;
        float sum = 0.0f;
        for (int i = 0; i < r1; i++)
            sum += w1b_sm[i * in_m + r] * gradZ_sm[i * r2 + j];
        gt1_ptr[b * (in_m * r2) + r * r2 + j] = sum;
    }
    __syncthreads();  // Ensure gt1 writes are visible

    // Phase 6: grad_x[b,r,s] = sum_j grad_temp1[r,j] * w2_b[j,s]
    for (int s = tid; s < in_n; s += bdim) {
        for (int r = 0; r < in_m; r++) {
            float sum = 0.0f;
            for (int j = 0; j < r2; j++)
                sum += gt1_ptr[b * (in_m * r2) + r * r2 + j] *
                       w2b_sm[j * in_n + s];
            write_dtype(grad_x_ptr, b * (in_m * in_n) + r * in_n + s,
                        sum, x_dtype);
        }
    }

    // Phase 7: grad_w2_a[k,j] += sum_i grad_temp2[b,i,k] * Z[b,i,j]  (atomicAdd)
    // grad_temp2 is in gt2_ptr (written in Phase 3, includes scalar)
    // Z is in Z_sm (computed in Phase 0b, forward values)
    // Target: out_k * r2 (small), accumulated across B blocks
    for (int idx = tid; idx < out_k * r2; idx += bdim) {
        int k = idx / r2;
        int j = idx % r2;
        float sum = 0.0f;
        for (int i = 0; i < r1; i++)
            sum += gt2_ptr[b * (r1 * out_k) + i * out_k + k] *
                   Z_sm[i * r2 + j];
        atomicAdd(&grad_w2a_ptr[k * r2 + j], sum);
    }

    // Phase 8: grad_w2_b[j,n] += sum_m grad_temp1[b,m,j] * X[b,m,n]  (atomicAdd)
    // grad_temp1 is in gt1_ptr (written in Phase 5, includes scalar through chain)
    // X read from global via read_dtype (handles bf16/fp16 natively)
    // Target: r2 * in_n (small), accumulated across B blocks
    for (int idx = tid; idx < r2 * in_n; idx += bdim) {
        int j = idx / in_n;
        int n = idx % in_n;
        float sum = 0.0f;
        for (int m = 0; m < in_m; m++)
            sum += gt1_ptr[b * (in_m * r2) + m * r2 + j] *
                   read_dtype(x_ptr, b * (in_m * in_n) + m * in_n + n, x_dtype);
        atomicAdd(&grad_w2b_ptr[j * in_n + n], sum);
    }
}

// ============================================================================
// Launcher functions (extern "C" interface for lokr_op.cpp)
// ============================================================================

static int compute_block_size(int n) {
    int bs = 1;
    while (bs * 2 <= n && bs * 2 <= 256) bs *= 2;
    return bs < 32 ? 32 : bs;
}

extern "C" {

void launch_kron1_fwd(
    const void* x_ptr, const void* w1_ptr, const void* w2_ptr,
    void* out_ptr,
    int B, int in_m, int in_n, int out_l, int out_k,
    float scalar,
    int x_dtype, int w1_dtype, int w2_dtype,
    cudaStream_t stream)
{
    int block = compute_block_size(out_k);
    int smem = (out_l * in_m + in_m * out_k) * sizeof(float);
    kron1_fwd_kernel<<<B, block, smem, stream>>>(
        x_ptr, w1_ptr, w2_ptr, out_ptr,
        B, in_m, in_n, out_l, out_k, scalar,
        x_dtype, w1_dtype, w2_dtype);
    CUDA_CHECK(cudaGetLastError());
}

void launch_kron1_bwd(
    const void* grad_out_ptr, const void* x_ptr,
    const void* w1_ptr, const void* w2_ptr,
    void* grad_x_ptr,
    float* grad_w1_ptr, float* grad_scalar_ptr, float* grad_temp_ptr,
    int B, int in_m, int in_n, int out_l, int out_k,
    float scalar,
    int grad_out_dtype, int x_dtype, int w1_dtype, int w2_dtype,
    cudaStream_t stream)
{
    int block = compute_block_size(out_k);
    int smem = (out_l * in_m + in_m * out_k) * sizeof(float);
    kron1_bwd_kernel<<<B, block, smem, stream>>>(
        grad_out_ptr, x_ptr, w1_ptr, w2_ptr,
        grad_x_ptr, grad_w1_ptr, grad_scalar_ptr, grad_temp_ptr,
        B, in_m, in_n, out_l, out_k, scalar,
        grad_out_dtype, x_dtype, w1_dtype, w2_dtype);
    CUDA_CHECK(cudaGetLastError());
}

void launch_kron2_fwd(
    const void* x_ptr,
    const void* w1_a_ptr, const void* w1_b_ptr,
    const void* w2_a_ptr, const void* w2_b_ptr,
    void* out_ptr, float* temp1_ptr,
    int B, int in_m, int in_n, int out_l, int out_k, int r1, int r2,
    float scalar,
    int x_dtype, int w1a_dtype, int w1b_dtype, int w2a_dtype, int w2b_dtype,
    cudaStream_t stream)
{
    int block = compute_block_size(out_k);
    int tz_size = in_m * r2 > r1 * r2 ? in_m * r2 : r1 * r2;
    int smem = (out_l * r1 + r1 * in_m + r2 * in_n + tz_size) * sizeof(float);
    kron2_fwd_kernel<<<B, block, smem, stream>>>(
        x_ptr, w1_a_ptr, w1_b_ptr, w2_a_ptr, w2_b_ptr,
        out_ptr, temp1_ptr,
        B, in_m, in_n, out_l, out_k, r1, r2, scalar,
        x_dtype, w1a_dtype, w1b_dtype, w2a_dtype, w2b_dtype);
    CUDA_CHECK(cudaGetLastError());
}

void launch_kron2_bwd(
    const void* grad_out_ptr,
    const float* temp1_ptr,
    const void* x_ptr,
    const void* w1_a_ptr, const void* w1_b_ptr,
    const void* w2_a_ptr, const void* w2_b_ptr,
    void* grad_x_ptr,
    float* grad_w1a_ptr, float* grad_w1b_ptr,
    float* grad_w2a_ptr, float* grad_w2b_ptr,
    float* grad_scalar_ptr,
    float* gt1_ptr, float* gt2_ptr,
    int B, int in_m, int in_n, int out_l, int out_k, int r1, int r2,
    float scalar,
    int grad_out_dtype, int x_dtype,
    int w1a_dtype, int w1b_dtype, int w2a_dtype, int w2b_dtype,
    cudaStream_t stream)
{
    int block = compute_block_size(out_k);
    // Shared: w1a + w1b + w2b + temp1 + Z + gradZ
    int smem = (out_l * r1 + r1 * in_m + r2 * in_n + in_m * r2 + r1 * r2 + r1 * r2) * sizeof(float);
    kron2_bwd_kernel<<<B, block, smem, stream>>>(
        grad_out_ptr, temp1_ptr, x_ptr,
        w1_a_ptr, w1_b_ptr, w2_a_ptr, w2_b_ptr,
        grad_x_ptr, grad_w1a_ptr, grad_w1b_ptr,
        grad_w2a_ptr, grad_w2b_ptr, grad_scalar_ptr,
        gt1_ptr, gt2_ptr,
        B, in_m, in_n, out_l, out_k, r1, r2, scalar,
        grad_out_dtype, x_dtype,
        w1a_dtype, w1b_dtype, w2a_dtype, w2b_dtype);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
