import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.lycoris_functional import make_kron, factorization, rebuild_tucker


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
            torch.nn.init.zeros_(self.lokr_w2)
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2_b)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        self.org_module_ref = [org_module]
        self._fused = False

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

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            diff_weight = self.make_weight(x.device).to(org_forwarded.dtype)
            diff_weight = diff_weight.view(self.shape)
            if self.org_module_ref[0].__class__.__name__ == "Conv2d":
                delta = F.conv2d(
                    x,
                    diff_weight,
                    None,
                    self.org_module_ref[0].stride,
                    self.org_module_ref[0].padding,
                )
            else:
                delta = F.linear(x, diff_weight)
            return org_forwarded + delta * self.multiplier

        if self._skip_module():
            return org_forwarded

        diff_weight = self.make_weight(x.device).float()
        diff_weight = diff_weight.view(self.shape)

        if self.org_module_ref[0].__class__.__name__ == "Conv2d":
            delta = F.conv2d(
                x.float(),
                diff_weight,
                None,
                self.org_module_ref[0].stride,
                self.org_module_ref[0].padding,
            )
        else:
            delta = F.linear(x.float(), diff_weight)

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
            destination["lokr_w1"] = self.lokr_w1
        else:
            destination["lokr_w1_a"] = self.lokr_w1_a
            destination["lokr_w1_b"] = self.lokr_w1_b
        if self.use_w2:
            destination["lokr_w2"] = self.lokr_w2
        else:
            destination["lokr_w2_a"] = self.lokr_w2_a
            destination["lokr_w2_b"] = self.lokr_w2_b
            if self.tucker:
                destination["lokr_t2"] = self.lokr_t2
        return destination
