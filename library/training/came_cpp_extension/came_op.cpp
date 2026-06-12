/**
 * CAME optimizer C++ dispatch layer.
 *
 * This file is compiled by MSVC. It allocates scratch buffers via ATen,
 * extracts raw float pointers, and calls extern "C" launcher functions
 * defined in came_cuda_kernel.cu (compiled by NVCC).
 *
 * Architecture:
 *   - All internal computation is in float32 (fp32)
 *   - State tensors are modified in-place by CUDA kernels
 *   - The param tensor may be bf16/fp16/fp32; P0 handles dtype conversion,
 *     F4/U2 handle dtype writeback
 *   - 0 ATen math operations — only torch::empty for scratch allocation
 *
 * Function signatures are fixed — do NOT change them. The Python wrapper
 * (__init__.py) calls these with specific argument ordering.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

// ============================================================================
// Forward declarations — extern "C" launchers from came_cuda_kernel.cu
// ============================================================================

extern "C" {
    void launch_prepare_param(
        const void* param_in, float* param_fp32_out, float* scalar_buf,
        int N, int scalar_buf_size, float wd_lr, int dtype,
        cudaStream_t stream);

    void launch_scale2(
        float* sq_col, float* res_col, int C,
        float beta1, float beta2, cudaStream_t stream);

    void launch_compute_sq_ema(
        const float* grad, float* sq_row, float* sq_col, float* scalar_buf,
        int R, int C, float beta1, float one_minus_beta1, float eps0,
        cudaStream_t stream);

    void launch_first_approx_and_norm(
        const float* grad, const float* sq_row, const float* sq_col,
        float* approx_buf, float* scalar_buf, int R, int C,
        cudaStream_t stream);

    void launch_clip_ema_residual_ema(
        const float* approx_buf, float* exp_avg,
        float* res_row, float* res_col, float* scalar_buf,
        int R, int C, float beta0, float one_minus_beta0,
        float beta2, float one_minus_beta2, float eps1, float clip_threshold,
        cudaStream_t stream);

    void launch_second_approx_and_apply(
        const float* res_row, const float* res_col,
        const float* exp_avg, const float* param_fp32, void* param_out,
        const float* scalar_buf, int R, int C, float lr, int dtype,
        cudaStream_t stream);

    void launch_unfactored_sq_rsqrt_and_norm(
        const float* grad, float* exp_avg_sq, float* buf, float* scalar_buf,
        int N, float beta1, float one_minus_beta1, float eps0,
        cudaStream_t stream);

    void launch_unfactored_clip_ema_apply(
        const float* buf, float* exp_avg,
        const float* param_fp32, void* param_out, const float* scalar_buf,
        int N, float beta0, float one_minus_beta0,
        float lr, float clip_threshold, int dtype,
        cudaStream_t stream);

    // --- Batched launchers (from came_cuda_batched.cu) ---

    void launch_batched_prepare_param(
        const void* param_in, float* param_fp32_out, float* scalar_buf,
        int total_N, int scalar_buf_size, float wd_lr, int dtype,
        cudaStream_t stream);

    void launch_batched_scale2(
        float* sq_col, float* res_col, int total_BC,
        float beta1, float beta2, cudaStream_t stream);

    void launch_batched_compute_sq_ema(
        const float* grad, float* sq_row, float* sq_col, float* scalar_buf,
        int B, int R, int C, float beta1, float one_minus_beta1, float eps0,
        cudaStream_t stream);

    void launch_batched_first_approx_and_norm(
        const float* grad, const float* sq_row, const float* sq_col,
        float* approx_buf, float* scalar_buf, int B, int R, int C,
        cudaStream_t stream);

    void launch_batched_clip_ema_residual_ema(
        const float* approx_buf, float* exp_avg,
        float* res_row, float* res_col, float* scalar_buf,
        int B, int R, int C, float beta0, float one_minus_beta0,
        float beta2, float one_minus_beta2, float eps1, float clip_threshold,
        cudaStream_t stream);

    void launch_batched_second_approx_and_apply(
        const float* res_row, const float* res_col,
        const float* exp_avg, const float* param_fp32, void* param_out,
        const float* scalar_buf, int B, int R, int C, float lr, int dtype,
        cudaStream_t stream);

    void launch_batched_unfactored_sq_rsqrt_and_norm(
        const float* grad, float* exp_avg_sq, float* buf, float* scalar_buf,
        int B, int N, float beta1, float one_minus_beta1, float eps0,
        cudaStream_t stream);

    void launch_batched_unfactored_clip_ema_apply(
        const float* buf, float* exp_avg,
        const float* param_fp32, void* param_out, const float* scalar_buf,
        int B, int N, float beta0, float one_minus_beta0,
        float lr, float clip_threshold, int dtype,
        cudaStream_t stream);
}

// Dtype flag constants matching came_cuda_kernel.cu
enum { DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_BF16 = 2 };

static int get_dtype_flag(const at::Tensor& t) {
    auto st = t.scalar_type();
    if (st == at::kBFloat16) return DTYPE_BF16;
    if (st == at::kHalf) return DTYPE_FP16;
    return DTYPE_FP32;
}

// Get raw void* from any-dtype tensor without triggering dtype dispatch checks.
// Uses storage().data() which returns void* regardless of dtype.
static void* raw_data_ptr(at::Tensor& t) {
    return static_cast<char*>(t.storage().mutable_data()) +
           static_cast<long long>(t.storage_offset()) * t.itemsize();
}

// ============================================================================
// Factored step for 2D+ parameters
// ============================================================================

void came_factored_step_cuda(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& exp_avg,
    at::Tensor& exp_avg_sq_row,
    at::Tensor& exp_avg_sq_col,
    at::Tensor& exp_avg_res_row,
    at::Tensor& exp_avg_res_col,
    float lr,
    float beta0,
    float beta1,
    float beta2,
    float eps0,
    float eps1,
    float clip_threshold,
    float weight_decay)
{
    TORCH_CHECK(grad.scalar_type() == at::kFloat,
        "CAME_C factored: grad must be float32");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat,
        "CAME_C factored: exp_avg must be float32");
    TORCH_CHECK(param.dim() >= 2,
        "CAME_C factored: param must be at least 2D");

    auto stream = c10::cuda::getCurrentCUDAStream();
    int R = static_cast<int>(grad.size(0));
    int C = static_cast<int>(grad.size(1));
    int dtype = get_dtype_flag(param);
    float wd_lr = weight_decay * lr;

    // Allocate scratch buffers (uninitialized — P0 clears scalar_buf)
    auto scalar_buf = torch::empty({3}, grad.options());
    auto approx_buf = torch::empty_like(grad);
    auto param_fp32 = torch::empty_like(grad);

    // Extract raw pointers
    float* grad_ptr = grad.data_ptr<float>();
    float* param_fp32_ptr = param_fp32.data_ptr<float>();
    float* exp_avg_ptr = exp_avg.data_ptr<float>();
    float* sq_row_ptr = exp_avg_sq_row.data_ptr<float>();
    float* sq_col_ptr = exp_avg_sq_col.data_ptr<float>();
    float* res_row_ptr = exp_avg_res_row.data_ptr<float>();
    float* res_col_ptr = exp_avg_res_col.data_ptr<float>();
    float* scalar_ptr = scalar_buf.data_ptr<float>();
    float* approx_ptr = approx_buf.data_ptr<float>();
    void* param_ptr = raw_data_ptr(param);

    // 6 CUDA launches, 0 ATen compute
    // P0: dtype conversion + weight decay + scalar_buf clear
    launch_prepare_param(param_ptr, param_fp32_ptr, scalar_ptr,
                         R * C, /*scalar_buf_size=*/3, wd_lr, dtype, stream);

    // S0: column state pre-multiply
    launch_scale2(sq_col_ptr, res_col_ptr, C, beta1, beta2, stream);

    // F1: grad^2+eps0, row/col EMA for exp_avg_sq
    launch_compute_sq_ema(grad_ptr, sq_row_ptr, sq_col_ptr, scalar_ptr,
                          R, C, beta1, 1.0f - beta1, eps0, stream);

    // F2: first approx_sq_grad * grad, L2 norm accumulation
    launch_first_approx_and_norm(grad_ptr, sq_row_ptr, sq_col_ptr,
                                 approx_ptr, scalar_ptr, R, C, stream);

    // F3: RMS clip, exp_avg EMA, residual row/col EMA
    launch_clip_ema_residual_ema(approx_ptr, exp_avg_ptr,
                                 res_row_ptr, res_col_ptr, scalar_ptr,
                                 R, C, beta0, 1.0f - beta0,
                                 beta2, 1.0f - beta2, eps1, clip_threshold,
                                 stream);

    // F4: second approx_sq_grad * exp_avg, param update + dtype writeback
    launch_second_approx_and_apply(res_row_ptr, res_col_ptr, exp_avg_ptr,
                                   param_fp32_ptr, param_ptr, scalar_ptr,
                                   R, C, lr, dtype, stream);
}

// ============================================================================
// Unfactored step for 1D parameters
// ============================================================================

void came_unfactored_step_cuda(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& exp_avg,
    at::Tensor& exp_avg_sq,
    float lr,
    float beta0,
    float beta1,
    float eps0,
    float clip_threshold,
    float weight_decay)
{
    TORCH_CHECK(grad.scalar_type() == at::kFloat,
        "CAME_C unfactored: grad must be float32");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat,
        "CAME_C unfactored: exp_avg must be float32");

    auto stream = c10::cuda::getCurrentCUDAStream();
    int N = static_cast<int>(grad.numel());
    int dtype = get_dtype_flag(param);
    float wd_lr = weight_decay * lr;

    // Allocate scratch (uninitialized — P0 clears scalar_buf)
    auto scalar_buf = torch::empty({1}, grad.options());
    auto buf = torch::empty_like(grad);
    auto param_fp32 = torch::empty_like(grad);

    // Extract raw pointers
    float* grad_ptr = grad.data_ptr<float>();
    float* param_fp32_ptr = param_fp32.data_ptr<float>();
    float* exp_avg_ptr = exp_avg.data_ptr<float>();
    float* exp_avg_sq_ptr = exp_avg_sq.data_ptr<float>();
    float* scalar_ptr = scalar_buf.data_ptr<float>();
    float* buf_ptr = buf.data_ptr<float>();
    void* param_ptr = raw_data_ptr(param);

    // 3 CUDA launches, 0 ATen compute
    // P0: dtype conversion + weight decay + scalar_buf clear
    launch_prepare_param(param_ptr, param_fp32_ptr, scalar_ptr,
                         N, /*scalar_buf_size=*/1, wd_lr, dtype, stream);

    // U1: exp_avg_sq EMA, rsqrt*grad, L2 norm
    launch_unfactored_sq_rsqrt_and_norm(grad_ptr, exp_avg_sq_ptr, buf_ptr,
                                        scalar_ptr, N,
                                        beta1, 1.0f - beta1, eps0, stream);

    // U2: RMS clip, exp_avg EMA, param update + dtype writeback
    launch_unfactored_clip_ema_apply(buf_ptr, exp_avg_ptr,
                                     param_fp32_ptr, param_ptr, scalar_ptr,
                                     N, beta0, 1.0f - beta0,
                                     lr, clip_threshold, dtype, stream);
}

// ============================================================================
// Batched factored step — processes B params of shape (R, C) in one go
// ============================================================================

void came_factored_batched_step_cuda(
    at::Tensor& param_stack,        // (B, R, C) — any dtype
    const at::Tensor& grad_stack,   // (B, R, C) — fp32
    at::Tensor& exp_avg_stack,      // (B, R, C) — fp32
    at::Tensor& sq_row_stack,       // (B, R) — fp32
    at::Tensor& sq_col_stack,       // (B, C) — fp32
    at::Tensor& res_row_stack,      // (B, R) — fp32
    at::Tensor& res_col_stack,      // (B, C) — fp32
    at::Tensor& scalar_buf,         // (B, 3) — fp32 pre-allocated
    at::Tensor& approx_buf,         // (B, R, C) — fp32 pre-allocated
    at::Tensor& param_fp32,         // (B, R, C) — fp32 pre-allocated
    float lr,
    float beta0,
    float beta1,
    float beta2,
    float eps0,
    float eps1,
    float clip_threshold,
    float weight_decay)
{
    TORCH_CHECK(grad_stack.scalar_type() == at::kFloat,
        "CAME_C batched factored: grad must be float32");
    TORCH_CHECK(exp_avg_stack.scalar_type() == at::kFloat,
        "CAME_C batched factored: exp_avg must be float32");
    TORCH_CHECK(param_stack.dim() == 3,
        "CAME_C batched factored: param must be 3D (B, R, C)");

    auto stream = c10::cuda::getCurrentCUDAStream();
    int B = static_cast<int>(param_stack.size(0));
    int R = static_cast<int>(param_stack.size(1));
    int C = static_cast<int>(param_stack.size(2));
    int dtype = get_dtype_flag(param_stack);
    float wd_lr = weight_decay * lr;

    // Extract raw pointers (all stacks are contiguous)
    float* grad_ptr = grad_stack.data_ptr<float>();
    float* param_fp32_ptr = param_fp32.data_ptr<float>();
    float* exp_avg_ptr = exp_avg_stack.data_ptr<float>();
    float* sq_row_ptr = sq_row_stack.data_ptr<float>();
    float* sq_col_ptr = sq_col_stack.data_ptr<float>();
    float* res_row_ptr = res_row_stack.data_ptr<float>();
    float* res_col_ptr = res_col_stack.data_ptr<float>();
    float* scalar_ptr = scalar_buf.data_ptr<float>();
    float* approx_ptr = approx_buf.data_ptr<float>();
    void* param_ptr = raw_data_ptr(param_stack);

    int total_N = B * R * C;
    int total_BC = B * C;

    // 6 batched CUDA launches, 0 ATen compute
    // P0: dtype conversion + weight decay + scalar_buf clear (B*3 entries)
    launch_batched_prepare_param(param_ptr, param_fp32_ptr, scalar_ptr,
                                 total_N, B * 3, wd_lr, dtype, stream);

    // S0: column state pre-multiply (B*C elements)
    launch_batched_scale2(sq_col_ptr, res_col_ptr, total_BC, beta1, beta2, stream);

    // F1: grad^2+eps0, row/col EMA
    launch_batched_compute_sq_ema(grad_ptr, sq_row_ptr, sq_col_ptr, scalar_ptr,
                                  B, R, C, beta1, 1.0f - beta1, eps0, stream);

    // F2: first approx_sq_grad * grad, L2 norm
    launch_batched_first_approx_and_norm(grad_ptr, sq_row_ptr, sq_col_ptr,
                                         approx_ptr, scalar_ptr, B, R, C, stream);

    // F3: RMS clip, exp_avg EMA, residual row/col EMA
    launch_batched_clip_ema_residual_ema(approx_ptr, exp_avg_ptr,
                                         res_row_ptr, res_col_ptr, scalar_ptr,
                                         B, R, C, beta0, 1.0f - beta0,
                                         beta2, 1.0f - beta2, eps1, clip_threshold,
                                         stream);

    // F4: second approx, param update + dtype writeback
    launch_batched_second_approx_and_apply(res_row_ptr, res_col_ptr, exp_avg_ptr,
                                           param_fp32_ptr, param_ptr, scalar_ptr,
                                           B, R, C, lr, dtype, stream);
}

// ============================================================================
// Batched unfactored step — processes B params of size N in one go
// ============================================================================

void came_unfactored_batched_step_cuda(
    at::Tensor& param_stack,        // (B, N) — any dtype
    const at::Tensor& grad_stack,   // (B, N) — fp32
    at::Tensor& exp_avg_stack,      // (B, N) — fp32
    at::Tensor& exp_avg_sq_stack,   // (B, N) — fp32
    at::Tensor& scalar_buf,         // (B,) — fp32 pre-allocated
    at::Tensor& buf,                // (B, N) — fp32 pre-allocated
    at::Tensor& param_fp32,         // (B, N) — fp32 pre-allocated
    float lr,
    float beta0,
    float beta1,
    float eps0,
    float clip_threshold,
    float weight_decay)
{
    TORCH_CHECK(grad_stack.scalar_type() == at::kFloat,
        "CAME_C batched unfactored: grad must be float32");
    TORCH_CHECK(exp_avg_stack.scalar_type() == at::kFloat,
        "CAME_C batched unfactored: exp_avg must be float32");

    auto stream = c10::cuda::getCurrentCUDAStream();
    int B = static_cast<int>(param_stack.size(0));
    int N = static_cast<int>(param_stack.size(1));
    int dtype = get_dtype_flag(param_stack);
    float wd_lr = weight_decay * lr;

    // Extract raw pointers
    float* grad_ptr = grad_stack.data_ptr<float>();
    float* param_fp32_ptr = param_fp32.data_ptr<float>();
    float* exp_avg_ptr = exp_avg_stack.data_ptr<float>();
    float* exp_avg_sq_ptr = exp_avg_sq_stack.data_ptr<float>();
    float* scalar_ptr = scalar_buf.data_ptr<float>();
    float* buf_ptr = buf.data_ptr<float>();
    void* param_ptr = raw_data_ptr(param_stack);

    // 3 batched CUDA launches
    // P0: dtype conversion + weight decay + scalar_buf clear (B entries)
    launch_batched_prepare_param(param_ptr, param_fp32_ptr, scalar_ptr,
                                 B * N, B, wd_lr, dtype, stream);

    // U1: exp_avg_sq EMA, rsqrt*grad, L2 norm
    launch_batched_unfactored_sq_rsqrt_and_norm(grad_ptr, exp_avg_sq_ptr, buf_ptr,
                                                scalar_ptr, B, N,
                                                beta1, 1.0f - beta1, eps0, stream);

    // U2: RMS clip, exp_avg EMA, param update + dtype writeback
    launch_batched_unfactored_clip_ema_apply(buf_ptr, exp_avg_ptr,
                                             param_fp32_ptr, param_ptr, scalar_ptr,
                                             B, N, beta0, 1.0f - beta0,
                                             lr, clip_threshold, dtype, stream);
}

// ============================================================================
// PyBind11 module definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("came_factored_step", &came_factored_step_cuda,
          "CAME factored update for 2D+ parameters (fused CUDA kernels)");
    m.def("came_unfactored_step", &came_unfactored_step_cuda,
          "CAME unfactored update for 1D parameters (fused CUDA kernels)");
    m.def("came_factored_batched_step", &came_factored_batched_step_cuda,
          "CAME batched factored update for B same-shape 2D+ parameters");
    m.def("came_unfactored_batched_step", &came_unfactored_batched_step_cuda,
          "CAME batched unfactored update for B same-shape 1D parameters");
}
