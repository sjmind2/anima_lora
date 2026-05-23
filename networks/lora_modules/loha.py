import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.lycoris_functional import HadaWeight, HadaWeightTucker


class LohaModule(BaseLoRAModule):
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

        self.tucker = False

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            self.shape = (out_dim, in_dim, *k_size)
            self.tucker = use_tucker and any(k != 1 for k in k_size)
            if self.tucker:
                w_shape = (out_dim, in_dim, *k_size)
            else:
                w_shape = (out_dim, in_dim * int(torch.tensor(k_size).prod().item()))
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.shape = (out_dim, in_dim)
            w_shape = (out_dim, in_dim)

        if self.tucker:
            self.hada_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w1_a = nn.Parameter(torch.empty(lora_dim, w_shape[0]))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))
            self.hada_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w2_a = nn.Parameter(torch.empty(lora_dim, w_shape[0]))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))
            self.hada_w2_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

        if self.tucker:
            torch.nn.init.normal_(self.hada_t1, std=0.1)
            torch.nn.init.normal_(self.hada_t2, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w2_b, std=1)
        torch.nn.init.zeros_(self.hada_w2_a)

        self.org_module_ref = [org_module]
        self._fused = False
        self.scalar = torch.tensor(1.0)

    def make_weight(self, device=None):
        scale = torch.tensor(self.scale, dtype=torch.float32, device=device)

        if self.tucker:
            w1b = self.hada_w1_b.to(device)
            w1a = self.hada_w1_a.to(device)
            w2b = self.hada_w2_b.to(device)
            w2a = self.hada_w2_a.to(device)
            t1 = self.hada_t1.to(device)
            t2 = self.hada_t2.to(device)
            weight = HadaWeightTucker.apply(t1, w1b, w1a, t2, w2b, w2a, scale)
        else:
            w1b = self.hada_w1_b.to(device)
            w1a = self.hada_w1_a.to(device)
            w2b = self.hada_w2_b.to(device)
            w2a = self.hada_w2_a.to(device)
            weight = HadaWeight.apply(w1b, w1a, w2b, w2a, scale)

        weight = weight * self.scalar.to(device)

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
            orig_norm = self.make_weight(device).norm()
            norm = torch.clamp(orig_norm, max_norm / 2)
            desired = torch.clamp(norm, max=max_norm)
            ratio = desired.cpu() / norm.cpu()
            scaled = norm != desired
            if scaled:
                self.scalar *= ratio
            return scaled, (orig_norm * ratio).item()

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            diff_weight = self.make_weight(x.device).to(org_forwarded.dtype)
            if self.org_module_ref[0].__class__.__name__ == "Conv2d":
                diff_weight = diff_weight.view(self.shape)
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

        diff_weight = self.make_weight(x.device).float()

        if self.org_module_ref[0].__class__.__name__ == "Conv2d":
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
            w1b = sd["hada_w1_b"].to(torch.float).to(device)
            w1a = sd["hada_w1_a"].to(torch.float).to(device)
            w2b = sd["hada_w2_b"].to(torch.float).to(device)
            w2a = sd["hada_w2_a"].to(torch.float).to(device)

            if "hada_t1" in sd and "hada_t2" in sd:
                t1 = sd["hada_t1"].to(torch.float).to(device)
                t2 = sd["hada_t2"].to(torch.float).to(device)
                scale = torch.tensor(self.scale, dtype=torch.float32, device=device)
                diff = HadaWeightTucker.apply(t1, w1b, w1a, t2, w2b, w2a, scale)
            else:
                scale = torch.tensor(self.scale, dtype=torch.float32, device=device)
                diff = HadaWeight.apply(w1b, w1a, w2b, w2a, scale)

            diff = diff.view(self.shape)
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
        destination["hada_w1_a"] = self.hada_w1_a * self.scalar.to(self.hada_w1_a.device)
        destination["hada_w1_b"] = self.hada_w1_b
        destination["hada_w2_a"] = self.hada_w2_a
        destination["hada_w2_b"] = self.hada_w2_b
        if self.tucker:
            destination["hada_t1"] = self.hada_t1
            destination["hada_t2"] = self.hada_t2
        return destination
