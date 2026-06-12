/**
 * LoKR forward/backward C++ dispatch layer.
 *
 * This file is compiled by MSVC. It allocates workspace via ATen,
 * extracts raw pointers, calls extern "C" CUDA launchers, and uses
 * ATen matmul (cuBLAS) for the cross-batch grad_w2 reduction.
 *
 * Architecture:
 *   - Custom CUDA kernels handle per-batch computation (temp, grad_x, grad_w1,
 *     grad_scalar, grad_temp materialization)
 *   - cuBLAS handles the cross-batch grad_w2 reduction (matmul of fp32 tensors)
 *   - All kernel computation in fp32; dtype conversion at read/write boundaries
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

// ============================================================================
// Forward declarations — extern "C" launchers from lokr_cuda_kernel.cu
// ============================================================================

extern "C" {
    void launch_kron1_fwd(
        const void* x_ptr, const void* w1_ptr, const void* w2_ptr,
        void* out_ptr,
        int B, int in_m, int in_n, int out_l, int out_k,
        float scalar,
        int x_dtype, int w1_dtype, int w2_dtype,
        cudaStream_t stream);

    void launch_kron1_bwd(
        const void* grad_out_ptr, const void* x_ptr,
        const void* w1_ptr, const void* w2_ptr,
        void* grad_x_ptr,
        float* grad_w1_ptr, float* grad_scalar_ptr, float* grad_temp_ptr,
        int B, int in_m, int in_n, int out_l, int out_k,
        float scalar,
        int grad_out_dtype, int x_dtype, int w1_dtype, int w2_dtype,
        cudaStream_t stream);

    void launch_kron2_fwd(
        const void* x_ptr,
        const void* w1_a_ptr, const void* w1_b_ptr,
        const void* w2_a_ptr, const void* w2_b_ptr,
        void* out_ptr, float* temp1_ptr,
        int B, int in_m, int in_n, int out_l, int out_k, int r1, int r2,
        float scalar,
        int x_dtype, int w1a_dtype, int w1b_dtype, int w2a_dtype, int w2b_dtype,
        cudaStream_t stream);

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
        cudaStream_t stream);
}

// Dtype flag constants matching lokr_cuda_kernel.cu
enum { DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_BF16 = 2 };

static int get_dtype_flag(const at::Tensor& t) {
    auto st = t.scalar_type();
    if (st == at::kBFloat16) return DTYPE_BF16;
    if (st == at::kHalf) return DTYPE_FP16;
    return DTYPE_FP32;
}

// Get raw void* from any-dtype tensor (handles non-zero storage_offset).
static const void* raw_const_ptr(const at::Tensor& t) {
    return static_cast<const char*>(t.storage().data()) +
           static_cast<long long>(t.storage_offset()) * t.itemsize();
}

static void* raw_mutable_ptr(at::Tensor& t) {
    return static_cast<char*>(t.storage().mutable_data()) +
           static_cast<long long>(t.storage_offset()) * t.itemsize();
}

// ============================================================================
// kron1_forward — Single-stage forward (cuBLAS GEMMs + fused scalar/cast)
//
// Uses cuBLAS bmm for the large GEMMs to leverage tensor cores.
// The naive scalar-loop kernel was 4x slower for in_n=1024 (down_proj).
// ============================================================================

at::Tensor kron1_forward(
    const at::Tensor& x,     // (*, in_m*in_n)
    const at::Tensor& w1,    // (out_l, in_m)
    const at::Tensor& w2,    // (out_k, in_n)
    double scalar)
{
    TORCH_CHECK(w1.dim() == 2, "w1 must be 2D");
    TORCH_CHECK(w2.dim() == 2, "w2 must be 2D");
    TORCH_CHECK(x.dim() >= 1, "x must be at least 1D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int out_l = static_cast<int>(w1.size(0));
    int in_m  = static_cast<int>(w1.size(1));
    int out_k = static_cast<int>(w2.size(0));
    int in_n  = static_cast<int>(w2.size(1));
    int feat  = in_m * in_n;
    int B     = static_cast<int>(x.numel()) / feat;

    TORCH_CHECK(x.numel() % feat == 0, "x last dim must equal in_m*in_n");

    // All computation in fp32 for numerical consistency with PyTorch reference
    auto x_fp32 = x.to(at::kFloat).view({B, in_m, in_n});
    auto w1_fp32 = w1.to(at::kFloat);
    auto w2_fp32 = w2.to(at::kFloat);

    // temp = X @ w2.T  → (B, in_m, out_k) via cuBLAS bmm
    auto w2_t = w2_fp32.t().unsqueeze(0).expand({B, in_n, out_k});
    auto temp = torch::bmm(x_fp32, w2_t);

    // result = einsum("pr,brk->bpk", w1, temp) → (B, out_l, out_k) via cuBLAS bmm
    auto w1_exp = w1_fp32.unsqueeze(0).expand({B, out_l, in_m});
    auto result = torch::bmm(w1_exp, temp);

    // Apply scalar and cast back to input dtype
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = out_l * out_k;
    auto out = (result.view({B, out_l * out_k}) * static_cast<float>(scalar))
                   .to(x.scalar_type())
                   .view(out_sizes);

    return out;
}

// ============================================================================
// kron1_backward — Single-stage backward (all cuBLAS)
//
// grad_w1/grad_w2/grad_x via cuBLAS, grad_scalar via reduction.
// No custom kernel — avoids the scalar-loop regression for large in_n.
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> kron1_backward(
    const at::Tensor& grad_out,  // (*, out_l*out_k)
    const at::Tensor& x,         // (*, in_m*in_n)
    const at::Tensor& w1,        // (out_l, in_m)
    const at::Tensor& w2,        // (out_k, in_n)
    double scalar)
{
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int out_l = static_cast<int>(w1.size(0));
    int in_m  = static_cast<int>(w1.size(1));
    int out_k = static_cast<int>(w2.size(0));
    int in_n  = static_cast<int>(w2.size(1));
    int feat  = in_m * in_n;
    int B     = static_cast<int>(x.numel()) / feat;

    // All fp32 for numerical consistency
    auto x_fp32 = x.to(at::kFloat).view({B, in_m, in_n});
    auto w1_fp32 = w1.to(at::kFloat);
    auto w2_fp32 = w2.to(at::kFloat);
    auto go_fp32 = grad_out.to(at::kFloat).view({B, out_l, out_k});

    // Recompute temp for grad_w1 and grad_scalar
    auto w2_t = w2_fp32.t().unsqueeze(0).expand({B, in_n, out_k});
    auto temp = torch::bmm(x_fp32, w2_t);  // (B, in_m, out_k)

    // fwd_out_raw = einsum("pr,brk->bpk", w1, temp)  (no scalar)
    auto w1_exp = w1_fp32.unsqueeze(0).expand({B, out_l, in_m});
    auto fwd_out_raw = torch::bmm(w1_exp, temp);  // (B, out_l, out_k)

    // grad_scalar = sum(grad_out * fwd_out_raw)
    auto grad_scalar = (go_fp32 * fwd_out_raw).sum();

    // grad_result = grad_out * scalar
    auto grad_result = go_fp32 * static_cast<float>(scalar);

    // grad_w1 = einsum("bpk,brk->pr", grad_result, temp) → (out_l, in_m)
    // Per batch: grad_result[b] @ temp[b].T = (out_l, out_k) @ (out_k, in_m)
    auto grad_w1_fp32 = torch::bmm(grad_result, temp.transpose(1, 2)).sum(0);

    // grad_temp = einsum("pr,bpk->brk", w1, grad_result)
    //           = w1.T @ grad_result per batch → (in_m, out_k)
    auto w1_t_exp = w1_fp32.t().unsqueeze(0).expand({B, in_m, out_l});
    auto grad_temp = torch::bmm(w1_t_exp, grad_result);  // (B, in_m, out_k)

    // grad_X = grad_temp @ w2  → (B, in_m, in_n)
    auto w2_exp = w2_fp32.unsqueeze(0).expand({B, out_k, in_n});
    auto grad_X = torch::bmm(grad_temp, w2_exp);

    // grad_w2 = einsum("brk,brs->ks", grad_temp, X) → (out_k, in_n)
    auto grad_w2_fp32 = grad_temp.view({-1, out_k}).t().matmul(x_fp32.view({-1, in_n}));

    // Reshape grad_x and cast
    auto grad_x = grad_X.view(x.sizes()).to(x.scalar_type());
    auto grad_w1_out = grad_w1_fp32.to(w1.scalar_type());
    auto grad_w2_out = grad_w2_fp32.to(w2.scalar_type());

    return std::make_tuple(grad_x, grad_w1_out, grad_w2_out, grad_scalar);
}

// ============================================================================
// kron2_forward — Two-stage fused forward (decompose_both)
// ============================================================================

std::tuple<at::Tensor, at::Tensor> kron2_forward(
    const at::Tensor& x,
    const at::Tensor& w1_a,   // (out_l, r1)
    const at::Tensor& w1_b,   // (r1, in_m)
    const at::Tensor& w2_a,   // (out_k, r2)
    const at::Tensor& w2_b,   // (r2, in_n)
    double scalar)
{
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int out_l = static_cast<int>(w1_a.size(0));
    int r1    = static_cast<int>(w1_a.size(1));
    int in_m  = static_cast<int>(w1_b.size(1));
    int out_k = static_cast<int>(w2_a.size(0));
    int r2    = static_cast<int>(w2_a.size(1));
    int in_n  = static_cast<int>(w2_b.size(1));
    int feat  = in_m * in_n;
    int B     = static_cast<int>(x.numel()) / feat;
    int out_feat = out_l * out_k;

    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out_sizes = x.sizes().vec();
    out_sizes.back() = out_feat;
    auto out = torch::empty(out_sizes, x.options());

    // temp1 saved for backward
    auto temp1 = torch::empty({B, in_m, r2},
        at::TensorOptions().dtype(at::kFloat).device(x.device()));

    launch_kron2_fwd(
        raw_const_ptr(x),
        raw_const_ptr(w1_a), raw_const_ptr(w1_b),
        raw_const_ptr(w2_a), raw_const_ptr(w2_b),
        raw_mutable_ptr(out), temp1.data_ptr<float>(),
        B, in_m, in_n, out_l, out_k, r1, r2,
        static_cast<float>(scalar),
        get_dtype_flag(x), get_dtype_flag(w1_a), get_dtype_flag(w1_b),
        get_dtype_flag(w2_a), get_dtype_flag(w2_b),
        stream);

    return std::make_tuple(out, temp1);
}

// ============================================================================
// kron2_backward — Two-stage fused backward
//   CUDA kernel: grad_x, grad_w1a (atomicAdd), grad_w1b (atomicAdd), grad_scalar,
//                grad_temp1, grad_temp2 materialization
//   cuBLAS:      grad_w2a = gt2.T @ Z,  grad_w2b = gt1.T @ x
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
kron2_backward(
    const at::Tensor& grad_out,
    const at::Tensor& temp1,   // saved from forward
    const at::Tensor& x,
    const at::Tensor& w1_a, const at::Tensor& w1_b,
    const at::Tensor& w2_a, const at::Tensor& w2_b,
    double scalar)
{
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(temp1.scalar_type() == at::kFloat, "temp1 must be fp32");

    int out_l = static_cast<int>(w1_a.size(0));
    int r1    = static_cast<int>(w1_a.size(1));
    int in_m  = static_cast<int>(w1_b.size(1));
    int out_k = static_cast<int>(w2_a.size(0));
    int r2    = static_cast<int>(w2_a.size(1));
    int in_n  = static_cast<int>(w2_b.size(1));
    int feat  = in_m * in_n;
    int B     = static_cast<int>(x.numel()) / feat;

    auto stream = c10::cuda::getCurrentCUDAStream();

    auto grad_x = torch::empty(x.sizes(), x.options());
    auto grad_w1a = torch::zeros({out_l, r1}, at::TensorOptions().dtype(at::kFloat).device(x.device()));
    auto grad_w1b = torch::zeros({r1, in_m}, at::TensorOptions().dtype(at::kFloat).device(x.device()));
    auto grad_w2a = torch::zeros({out_k, r2}, at::TensorOptions().dtype(at::kFloat).device(x.device()));
    auto grad_w2b = torch::zeros({r2, in_n}, at::TensorOptions().dtype(at::kFloat).device(x.device()));
    auto grad_scalar = torch::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

    // Workspaces
    auto gt1 = torch::empty({B, in_m, r2}, at::TensorOptions().dtype(at::kFloat).device(x.device()));
    auto gt2 = torch::empty({B, r1, out_k}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

    // All gradients computed in-kernel — no cuBLAS calls needed
    launch_kron2_bwd(
        raw_const_ptr(grad_out),
        temp1.data_ptr<float>(),
        raw_const_ptr(x),
        raw_const_ptr(w1_a), raw_const_ptr(w1_b),
        raw_const_ptr(w2_a), raw_const_ptr(w2_b),
        raw_mutable_ptr(grad_x),
        grad_w1a.data_ptr<float>(), grad_w1b.data_ptr<float>(),
        grad_w2a.data_ptr<float>(), grad_w2b.data_ptr<float>(),
        grad_scalar.data_ptr<float>(),
        gt1.data_ptr<float>(), gt2.data_ptr<float>(),
        B, in_m, in_n, out_l, out_k, r1, r2,
        static_cast<float>(scalar),
        get_dtype_flag(grad_out), get_dtype_flag(x),
        get_dtype_flag(w1_a), get_dtype_flag(w1_b),
        get_dtype_flag(w2_a), get_dtype_flag(w2_b),
        stream);

    // Cast gradients to parameter dtypes
    auto grad_w1a_out = grad_w1a.to(w1_a.scalar_type());
    auto grad_w1b_out = grad_w1b.to(w1_b.scalar_type());
    auto grad_w2a_out = grad_w2a.to(w2_a.scalar_type());
    auto grad_w2b_out = grad_w2b.to(w2_b.scalar_type());

    return std::make_tuple(grad_x, grad_w1a_out, grad_w1b_out,
                           grad_w2a_out, grad_w2b_out, grad_scalar);
}

// ============================================================================
// PyBind11 module definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kron1_forward", &kron1_forward,
          "LoKR single-stage forward (cuBLAS GEMMs)");
    m.def("kron1_backward", &kron1_backward,
          "LoKR single-stage backward (cuBLAS GEMMs)");
    m.def("kron2_forward", &kron2_forward,
          "LoKR two-stage fused forward (CUDA kernel)");
    m.def("kron2_backward", &kron2_backward,
          "LoKR two-stage fused backward (fully fused CUDA kernel)");
}
