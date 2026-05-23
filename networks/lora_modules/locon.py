import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.lycoris_functional import rebuild_tucker


class LoConModule(BaseLoRAModule):
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
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.shape = (out_dim, in_dim, *kernel_size)
            use_tucker = use_tucker and any(k != 1 for k in kernel_size)
            if use_tucker:
                self.lora_down = nn.Conv2d(
                    in_dim, self.lora_dim, 1, bias=False
                )
                self.lora_mid = nn.Conv2d(
                    self.lora_dim, self.lora_dim, kernel_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.lora_down = nn.Conv2d(
                    in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
                )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, 1, bias=False)
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.shape = (out_dim, in_dim)
            self.lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        if self.tucker:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        self.org_module_ref = [org_module]
        self._fused = False
        self.scalar = torch.tensor(1.0)

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        if self.tucker:
            t = self.lora_mid.weight.to(device)
            wa = wa.view(wa.size(0), -1).transpose(0, 1)
            wb = wb.view(wb.size(0), -1)
            weight = rebuild_tucker(t, wa, wb)
        else:
            weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
        weight = weight.view(self.shape)
        return weight

    def apply_max_norm(self, max_norm, device=None):
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            orig_norm = self.make_weight(device).norm() * self.scale
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
                delta = F.conv2d(
                    x,
                    diff_weight,
                    None,
                    self.org_module_ref[0].stride,
                    self.org_module_ref[0].padding,
                )
            else:
                delta = F.linear(x, diff_weight)
            return org_forwarded + delta * self.multiplier * self.scale * self.scalar.to(x.device)

        if self._skip_module():
            return org_forwarded

        diff_weight = self.make_weight(x.device).float()

        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=x.device) > self.rank_dropout).to(
                diff_weight.dtype
            )
            drop = drop.view(-1, *[1] * len(diff_weight.shape[1:]))
            diff_weight = diff_weight * drop

        if self.org_module_ref[0].__class__.__name__ == "Conv2d":
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

        return org_forwarded + (delta * self.multiplier * self.scale * self.scalar.to(x.device)).to(org_forwarded.dtype)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        weight = self.make_weight().to(torch.float)
        return multiplier * weight * self.scale * self.scalar.to(weight.device)

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module_ref[0].weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()

            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            if "lora_mid.weight" in sd and self.tucker:
                mid_weight = sd["lora_mid.weight"].to(torch.float).to(device)
                up_w = up_weight.view(up_weight.size(0), -1).transpose(0, 1)
                down_w = down_weight.view(down_weight.size(0), -1)
                diff = rebuild_tucker(mid_weight, up_w, down_w).view(self.shape)
            elif len(down_weight.size()) == 2:
                diff = up_weight @ down_weight
            elif down_weight.size()[2:4] == (1, 1):
                diff = (
                    (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            else:
                diff = F.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)

            w += self.multiplier * diff * self.scale * self.scalar.to(device)
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
        destination["lora_up.weight"] = self.lora_up.weight * self.scalar.to(self.lora_up.weight.device)
        destination["lora_down.weight"] = self.lora_down.weight
        if self.tucker:
            destination["lora_mid.weight"] = self.lora_mid.weight
        return destination
