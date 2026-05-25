import torch
import sys
sys.path.insert(0, '.')

from library.training.came_optimizer import CAME as PortedCAME
from came_pytorch import CAME as OriginalCAME

torch.manual_seed(42)
lr = 1e-4
betas = (0.9, 0.999, 0.9999)
eps = (1e-30, 1e-16)

# ========== Test 1: 2D factored ==========
print("=" * 60)
print("Test 1: 2D factored (10 steps)")
print("=" * 60)
torch.manual_seed(100)
p1 = torch.randn(16, 64, requires_grad=True)
p2 = p1.clone().detach().requires_grad_(True)

opt1 = PortedCAME([p1], lr=lr, betas=betas, eps=eps)
opt2 = OriginalCAME([p2], lr=lr, betas=betas, eps=eps)

for step in range(10):
    torch.manual_seed(1000 + step)
    grad = torch.randn(16, 64)
    p1.grad = grad.clone()
    p2.grad = grad.clone()
    opt1.step()
    opt2.step()

diff_2d = (p1 - p2).abs().max().item()
print(f"Ported param mean: {p1.mean().item():.8f}")
print(f"Original param mean: {p2.mean().item():.8f}")
print(f"Max abs diff: {diff_2d}")
print(f"Result: {'PASS' if diff_2d < 1e-6 else 'FAIL'}")

# ========== Test 2: 1D unfactored ==========
print()
print("=" * 60)
print("Test 2: 1D unfactored (10 steps)")
print("=" * 60)
torch.manual_seed(200)
p3 = torch.randn(128, requires_grad=True)
p4 = p3.clone().detach().requires_grad_(True)

opt3 = PortedCAME([p3], lr=lr, betas=betas, eps=eps)
opt4 = OriginalCAME([p4], lr=lr, betas=betas, eps=eps)

for step in range(10):
    torch.manual_seed(2000 + step)
    grad = torch.randn(128)
    p3.grad = grad.clone()
    p4.grad = grad.clone()
    opt3.step()
    opt4.step()

diff_1d = (p3 - p4).abs().max().item()
print(f"Ported param mean: {p3.mean().item():.8f}")
print(f"Original param mean: {p4.mean().item():.8f}")
print(f"Max abs diff: {diff_1d}")
print(f"Result: {'PASS' if diff_1d < 1e-6 else 'FAIL'}")

# ========== Test 3: 2D with weight_decay ==========
print()
print("=" * 60)
print("Test 3: 2D with weight_decay=0.01 (10 steps)")
print("=" * 60)
torch.manual_seed(300)
p5 = torch.randn(16, 64, requires_grad=True)
p6 = p5.clone().detach().requires_grad_(True)

opt5 = PortedCAME([p5], lr=lr, betas=betas, eps=eps, weight_decay=0.01)
opt6 = OriginalCAME([p6], lr=lr, betas=betas, eps=eps, weight_decay=0.01)

for step in range(10):
    torch.manual_seed(3000 + step)
    grad = torch.randn(16, 64)
    p5.grad = grad.clone()
    p6.grad = grad.clone()
    opt5.step()
    opt6.step()

diff_wd = (p5 - p6).abs().max().item()
print(f"Ported param mean: {p5.mean().item():.8f}")
print(f"Original param mean: {p6.mean().item():.8f}")
print(f"Max abs diff: {diff_wd}")
print(f"Result: {'PASS' if diff_wd < 1e-6 else 'FAIL'}")

# ========== Test 4: bf16 gradient ==========
print()
print("=" * 60)
print("Test 4: bf16 gradient (10 steps)")
print("=" * 60)
torch.manual_seed(400)
p7 = torch.randn(16, 64, requires_grad=True)
p8 = p7.clone().detach().requires_grad_(True)

opt7 = PortedCAME([p7], lr=lr, betas=betas, eps=eps)
opt8 = OriginalCAME([p8], lr=lr, betas=betas, eps=eps)

for step in range(10):
    torch.manual_seed(4000 + step)
    grad = torch.randn(16, 64, dtype=torch.bfloat16).float()
    p7.grad = grad.clone()
    p8.grad = grad.clone()
    opt7.step()
    opt8.step()

diff_bf = (p7 - p8).abs().max().item()
print(f"Ported param mean: {p7.mean().item():.8f}")
print(f"Original param mean: {p8.mean().item():.8f}")
print(f"Max abs diff: {diff_bf}")
print(f"Result: {'PASS' if diff_bf < 1e-6 else 'FAIL'}")

# ========== Summary ==========
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
all_pass = diff_2d < 1e-6 and diff_1d < 1e-6 and diff_wd < 1e-6 and diff_bf < 1e-6
print(f"2D factored: {'PASS' if diff_2d < 1e-6 else 'FAIL'} (diff={diff_2d})")
print(f"1D unfactored: {'PASS' if diff_1d < 1e-6 else 'FAIL'} (diff={diff_1d})")
print(f"2D + weight_decay: {'PASS' if diff_wd < 1e-6 else 'FAIL'} (diff={diff_wd})")
print(f"bf16 gradient: {'PASS' if diff_bf < 1e-6 else 'FAIL'} (diff={diff_bf})")
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
