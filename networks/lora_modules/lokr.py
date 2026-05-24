import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.lycoris_functional import make_kron, factorization, rebuild_tucker


class KronLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, scalar):
        out_l, in_m = w1.shape
        out_k, in_n = w2.shape
        leading = x.shape[:-1]
        x_f = x.float().reshape(-1, in_m * in_n)
        w1_f = w1.float()
        w2_f = w2.float()
        X = x_f.reshape(x_f.shape[0], in_m, in_n)
        temp = X @ w2_f.T
        result = torch.einsum("pr,brk->bpk", w1_f, temp)
        out = result.reshape(x_f.shape[0], out_l * out_k)
        out = (out * scalar).to(x.dtype)
        out = out.reshape(*leading, out_l * out_k)
        ctx.save_for_backward(x, w1, w2, scalar)
        ctx._leading = leading
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2, scalar = ctx.saved_tensors
        leading = ctx._leading
        out_l, in_m = w1.shape
        out_k, in_n = w2.shape
        x_f = x.float().reshape(-1, in_m * in_n)
        w1_f = w1.float()
        w2_f = w2.float()
        X = x_f.reshape(x_f.shape[0], in_m, in_n)
        temp = X @ w2_f.T
        result = torch.einsum("pr,brk->bpk", w1_f, temp)
        fwd_out = result.reshape(x_f.shape[0], out_l * out_k)
        go = grad_out.float().reshape(-1, out_l * out_k)
        grad_fwd = go * scalar
        grad_scalar = (go * fwd_out).sum()
        grad_result = grad_fwd.reshape(x_f.shape[0], out_l, out_k)
        grad_w1 = torch.einsum("bpk,brk->pr", grad_result, temp)
        grad_temp = torch.einsum("pr,bpk->brk", w1_f, grad_result)
        grad_X = grad_temp @ w2_f
        grad_w2 = torch.einsum("brk,brs->ks", grad_temp, X)
        grad_x = grad_X.reshape(x.shape)
        return grad_x.to(x.dtype), grad_w1.to(w1.dtype), grad_w2.to(w2.dtype), grad_scalar.to(scalar.dtype)


class KronLinearTwoStageFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_a, w1_b, w2_a, w2_b, scalar):
        r1 = w1_b.shape[0]
        r2 = w2_b.shape[0]
        leading = x.shape[:-1]
        in_m = w1_b.shape[1]
        in_n = w2_b.shape[1]
        x_f = x.float().reshape(-1, in_m * in_n)
        w1b_f = w1_b.float()
        w2b_f = w2_b.float()
        w1a_f = w1_a.float()
        w2a_f = w2_a.float()
        X = x_f.reshape(x_f.shape[0], in_m, in_n)
        temp1 = X @ w2b_f.T
        result1 = torch.einsum("pr,brk->bpk", w1b_f, temp1)
        z = result1.reshape(x_f.shape[0], r1 * r2)
        in_m_a = w1_a.shape[1]
        in_n_a = w2_a.shape[1]
        Z = z.reshape(x_f.shape[0], in_m_a, in_n_a)
        temp2 = Z @ w2a_f.T
        result2 = torch.einsum("pr,brk->bpk", w1a_f, temp2)
        out = result2.reshape(x_f.shape[0], w1_a.shape[0] * w2_a.shape[0])
        out = (out * scalar).to(x.dtype)
        out = out.reshape(*leading, w1_a.shape[0] * w2_a.shape[0])
        ctx.save_for_backward(x, w1_a, w1_b, w2_a, w2_b, scalar)
        ctx._leading = leading
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w1_a, w1_b, w2_a, w2_b, scalar = ctx.saved_tensors
        leading = ctx._leading
        r1 = w1_b.shape[0]
        r2 = w2_b.shape[0]
        in_m = w1_b.shape[1]
        in_n = w2_b.shape[1]
        out_l_a = w1_a.shape[0]
        out_k_a = w2_a.shape[0]
        in_m_a = w1_a.shape[1]
        in_n_a = w2_a.shape[1]
        x_f = x.float().reshape(-1, in_m * in_n)
        w1b_f = w1_b.float()
        w2b_f = w2_b.float()
        w1a_f = w1_a.float()
        w2a_f = w2_a.float()
        X = x_f.reshape(x_f.shape[0], in_m, in_n)
        temp1 = X @ w2b_f.T
        result1 = torch.einsum("pr,brk->bpk", w1b_f, temp1)
        z = result1.reshape(x_f.shape[0], r1 * r2)
        Z = z.reshape(x_f.shape[0], in_m_a, in_n_a)
        temp2 = Z @ w2a_f.T
        result2 = torch.einsum("pr,brk->bpk", w1a_f, temp2)
        fwd_out = result2.reshape(x_f.shape[0], out_l_a * out_k_a)
        go = grad_out.float().reshape(-1, out_l_a * out_k_a)
        grad_fwd2 = go * scalar
        grad_scalar = (go * fwd_out).sum()
        grad_result2 = grad_fwd2.reshape(x_f.shape[0], out_l_a, out_k_a)
        grad_w1a = torch.einsum("bpk,brk->pr", grad_result2, temp2)
        grad_temp2 = torch.einsum("pr,bpk->brk", w1a_f, grad_result2)
        grad_Z = grad_temp2 @ w2a_f
        grad_w2a = torch.einsum("brk,brs->ks", grad_temp2, Z)
        grad_z = grad_Z.reshape(x_f.shape[0], r1 * r2)
        grad_result1 = grad_z.reshape(x_f.shape[0], r1, r2)
        grad_w1b = torch.einsum("bpk,brk->pr", grad_result1, temp1)
        grad_temp1 = torch.einsum("pr,bpk->brk", w1b_f, grad_result1)
        grad_X = grad_temp1 @ w2b_f
        grad_w2b = torch.einsum("brk,brs->ks", grad_temp1, X)
        grad_x = grad_X.reshape(x.shape)
        return (
            grad_x.to(x.dtype),
            grad_w1a.to(w1_a.dtype),
            grad_w1b.to(w1_b.dtype),
            grad_w2a.to(w2_a.dtype),
            grad_w2b.to(w2_b.dtype),
            grad_scalar.to(scalar.dtype),
        )


class LokrModule(BaseLoRAModule):
    supports_conv2d = True

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
        use_tucker=False,
        decompose_both=False,
        lokr_factor=-1,
        full_matrix=False,
        use_scalar=False,
        weight_decompose=False,
        min_out_l=1,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        factor = int(lokr_factor)
        self.use_w1 = False
        self.use_w2 = False
        self.tucker = False
        self.full_matrix = full_matrix

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            self.shape = (out_dim, in_dim, *k_size)

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            if out_l < min_out_l and out_k >= min_out_l:
                out_l, out_k = out_k, out_l
            shape = ((out_l, out_k), (in_m, in_n), *k_size)
            self.tucker = use_tucker and any(k != 1 for k in k_size)

            if (
                decompose_both
                and lora_dim < max(shape[0][0], shape[1][0]) / 2
                and not self.full_matrix
            ):
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))

            if lora_dim >= max(shape[0][1], shape[1][1]) / 2 or self.full_matrix:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1], *k_size))
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *shape[2:]))
                self.lokr_w2_a = nn.Parameter(torch.empty(lora_dim, shape[0][1]))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
            else:
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1] * int(torch.tensor(shape[2:]).prod().item()))
                )
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.shape = (out_dim, in_dim)

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            if out_l < min_out_l and out_k >= min_out_l:
                out_l, out_k = out_k, out_l
            shape = ((out_l, out_k), (in_m, in_n))

            if (
                decompose_both
                and lora_dim < max(shape[0][0], shape[1][0]) / 2
                and not self.full_matrix
            ):
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))

            if lora_dim < max(shape[0][1], shape[1][1]) / 2 and not self.full_matrix:
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))

        if self.use_w2:
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
            else:
                torch.nn.init.zeros_(self.lokr_w2)
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            else:
                torch.nn.init.zeros_(self.lokr_w2_b)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        self.org_module_ref = [org_module]
        self._fused = False

        if self.use_w1 and self.use_w2:
            self.scale = 1.0
            self.alpha = torch.tensor(float(self.lora_dim))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        self.wd = weight_decompose
        if weight_decompose:
            org_weight = org_module.weight.data.clone().float()
            out_dim = org_weight.shape[0]
            if org_module.__class__.__name__ == "Conv2d":
                self.dora_norm_dims = 2
                self.dora_scale = nn.Parameter(
                    torch.norm(org_weight.reshape(out_dim, -1), dim=1, keepdim=True)
                    .reshape(out_dim, *[1] * (org_weight.dim() - 1))
                )
            else:
                self.dora_norm_dims = 0
                self.dora_scale = nn.Parameter(
                    torch.norm(org_weight, dim=1, keepdim=True)
                )
        else:
            self.dora_norm_dims = 0

    def make_weight(self, device=None):
        w1 = (self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b).to(device)

        if self.use_w2:
            w2 = self.lokr_w2.to(device)
        elif self.tucker:
            w2 = rebuild_tucker(
                self.lokr_t2.to(device),
                self.lokr_w2_a.to(device),
                self.lokr_w2_b.to(device),
            )
        else:
            w2_a = self.lokr_w2_a.to(device)
            w2_b = self.lokr_w2_b.to(device)
            if w2_b.dim() > 2:
                r, o, *k = w2_b.shape
                w2 = (w2_a @ w2_b.view(r, -1)).view(-1, o, *k)
            else:
                w2 = w2_a @ w2_b

        weight = make_kron(w1, w2, self.scale)

        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(weight.size(0), device=device) > self.rank_dropout).to(
                weight.dtype
            )
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            weight = weight * drop

        return weight

    def apply_max_norm(self, max_norm, device=None):
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            if self.use_w1:
                w1_n = self.lokr_w1.to(device).norm()
            else:
                w1_n = self.lokr_w1_a.to(device).norm() * self.lokr_w1_b.to(device).norm()
            if self.use_w2:
                w2_n = self.lokr_w2.to(device).norm()
            elif self.tucker:
                t2 = self.lokr_t2.to(device)
                w2a = self.lokr_w2_a.to(device)
                w2b = self.lokr_w2_b.to(device)
                w2_n = t2.norm() * w2a.norm() * w2b.norm()
            else:
                w2_n = self.lokr_w2_a.to(device).norm() * self.lokr_w2_b.to(device).norm()
            upper_bound = w1_n * w2_n * abs(self.scale) * self.scalar.to(device).abs()
            if upper_bound.item() <= max_norm:
                return False, upper_bound.item()
            orig_norm = self.make_weight(device).norm()
            norm = torch.clamp(orig_norm, max_norm / 2)
            desired = torch.clamp(norm, max=max_norm)
            ratio = desired.cpu() / norm.cpu()
            scaled = norm != desired
            if scaled:
                modules = 4 - self.use_w1 - self.use_w2 + (not self.use_w2 and self.tucker)
                r = ratio ** (1 / modules)
                if self.use_w1:
                    self.lokr_w1 *= r
                else:
                    self.lokr_w1_a *= r
                    self.lokr_w1_b *= r
                if self.use_w2:
                    self.lokr_w2 *= r
                else:
                    if self.tucker:
                        self.lokr_t2 *= r
                    self.lokr_w2_a *= r
                    self.lokr_w2_b *= r
            return scaled, (orig_norm * ratio).item()

    def apply_weight_decompose(self, weight, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        if self.dora_norm_dims == 2:
            weight_norm = (
                torch.norm(weight.reshape(weight.shape[0], -1), dim=1, keepdim=True)
                .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
            ) + torch.finfo(weight.dtype).eps
        else:
            weight_norm = torch.norm(weight, dim=1, keepdim=True) + torch.finfo(weight.dtype).eps
        scale = self.dora_scale / weight_norm
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1
        return weight * scale

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            diff_weight = self.make_weight(x.device).to(org_forwarded.dtype) * self.scalar
            diff_weight = diff_weight.view(self.shape)
            if self.wd:
                base_weight = self.org_module_ref[0].weight.data.to(x.device).float()
                new_weight = self.apply_weight_decompose(base_weight + diff_weight.float())
                delta_weight = new_weight - base_weight
                delta_weight = delta_weight.to(org_forwarded.dtype)
                if self.org_module_ref[0].__class__.__name__ == "Conv2d":
                    delta = F.conv2d(
                        x,
                        delta_weight,
                        None,
                        self.org_module_ref[0].stride,
                        self.org_module_ref[0].padding,
                        self.org_module_ref[0].dilation,
                        self.org_module_ref[0].groups,
                    )
                else:
                    delta = F.linear(x, delta_weight)
                return org_forwarded + delta
            if self.org_module_ref[0].__class__.__name__ == "Conv2d":
                delta = F.conv2d(
                    x,
                    diff_weight,
                    None,
                    self.org_module_ref[0].stride,
                    self.org_module_ref[0].padding,
                    self.org_module_ref[0].dilation,
                    self.org_module_ref[0].groups,
                )
            else:
                delta = F.linear(x, diff_weight)
            return org_forwarded + delta * self.multiplier

        if self._skip_module():
            return org_forwarded

        if self.wd:
            diff_weight = self.make_weight(x.device).float() * self.scalar
            diff_weight = diff_weight.view(self.shape)
            base_weight = self.org_module_ref[0].weight.data.to(x.device).float()
            new_weight = self.apply_weight_decompose(base_weight + diff_weight)
            delta_weight = new_weight - base_weight
            if self.org_module_ref[0].__class__.__name__ == "Conv2d":
                delta = F.conv2d(
                    x.float(),
                    delta_weight,
                    None,
                    self.org_module_ref[0].stride,
                    self.org_module_ref[0].padding,
                    self.org_module_ref[0].dilation,
                    self.org_module_ref[0].groups,
                )
            else:
                delta = F.linear(x.float(), delta_weight)
            return org_forwarded + delta.to(org_forwarded.dtype)

        if self.org_module_ref[0].__class__.__name__ == "Conv2d":
            diff_weight = self.make_weight(x.device).float() * self.scalar
            diff_weight = diff_weight.view(self.shape)
            delta = F.conv2d(
                x.float(),
                diff_weight,
                None,
                self.org_module_ref[0].stride,
                self.org_module_ref[0].padding,
                self.org_module_ref[0].dilation,
                self.org_module_ref[0].groups,
            )
        else:
            if not self.use_w1 and not self.use_w2:
                delta = KronLinearTwoStageFn.apply(
                    x, self.lokr_w1_a, self.lokr_w1_b,
                    self.lokr_w2_a, self.lokr_w2_b, self.scalar * self.scale,
                )
            else:
                if self.use_w1:
                    w1 = self.lokr_w1
                else:
                    w1 = self.lokr_w1_a @ self.lokr_w1_b
                if self.use_w2:
                    w2 = self.lokr_w2
                else:
                    w2 = self.lokr_w2_a @ self.lokr_w2_b
                delta = KronLinearFn.apply(x, w1, w2, self.scalar * self.scale)
            if self.rank_dropout is not None and self.training:
                drop = (torch.rand(delta.shape[-1], device=x.device) > self.rank_dropout).to(
                    delta.dtype
                )
                delta = delta * drop

        return org_forwarded + (delta * self.multiplier).to(org_forwarded.dtype)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        weight = self.make_weight().to(torch.float)
        return multiplier * weight

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module_ref[0].weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()

            if "lokr_w1" in sd:
                w1 = sd["lokr_w1"].to(torch.float).to(device)
            else:
                w1 = sd["lokr_w1_a"].to(torch.float).to(device) @ sd["lokr_w1_b"].to(torch.float).to(device)

            if "lokr_w2" in sd:
                w2 = sd["lokr_w2"].to(torch.float).to(device)
            elif "lokr_t2" in sd:
                w2 = rebuild_tucker(
                    sd["lokr_t2"].to(torch.float).to(device),
                    sd["lokr_w2_a"].to(torch.float).to(device),
                    sd["lokr_w2_b"].to(torch.float).to(device),
                )
            else:
                w2_a = sd["lokr_w2_a"].to(torch.float).to(device)
                w2_b = sd["lokr_w2_b"].to(torch.float).to(device)
                if w2_b.dim() > 2:
                    r, o, *k = w2_b.shape
                    w2 = (w2_a @ w2_b.view(r, -1)).view(-1, o, *k)
                else:
                    w2 = w2_a @ w2_b

            diff = make_kron(w1, w2, self.scale).view(self.shape)
            w += self.multiplier * diff
            weight.data.copy_(w.to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        if self.use_w1:
            destination["lokr_w1"] = self.lokr_w1 * self.scalar
        else:
            destination["lokr_w1_a"] = self.lokr_w1_a * self.scalar
            destination["lokr_w1_b"] = self.lokr_w1_b
        if self.use_w2:
            destination["lokr_w2"] = self.lokr_w2
        else:
            destination["lokr_w2_a"] = self.lokr_w2_a
            destination["lokr_w2_b"] = self.lokr_w2_b
            if self.tucker:
                destination["lokr_t2"] = self.lokr_t2
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        return destination
