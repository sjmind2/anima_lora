# HydraLoRA: MoE-style multi-head LoRA with layer-local routing.

import math
from typing import List, Optional

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.custom_autograd import lora_down_project
from networks.lora_modules.router_state import (
    _apply_sigma_band_mask,
    _clear_fei_feature_cache,
    _clear_routing_weights,
    _clear_sigma_feature_cache,
    _register_fei_feature_cache,
    _register_routing_weights_buffer,
    _register_sigma_band_partition,
    _register_sigma_feature_cache,
    _set_fei_feature_cache,
    _set_routing_weights,
    _set_sigma_feature_cache,
    _sigma_sinusoidal_features,
)

# Re-exported through ``networks.lora_modules.__init__``; ``network.py``
# imports ``_sigma_sinusoidal_features`` from there.
__all__ = [
    "HydraLoRAModule",
    "_apply_sigma_band_mask",
    "_sigma_sinusoidal_features",
]


class HydraLoRAModule(BaseLoRAModule):
    """HydraLoRA: shared lora_down + per-expert lora_up, layer-local routing.

    See docs/methods/hydra-lora.md.

    Routing inputs (concatenated into the router's input):
      * pooled rank-R signal (always)
      * sinusoidal(σ) when ``sigma_feature_dim > 0``
      * FEI(z_t) when ``fei_feature_dim > 0``

    σ goes into the router *input* (not as additive bias) so its gradient
    survives even when ``score_e`` is near-uniform — a bias-only path's
    ``dL/d logits · d_sigma_feat`` vanishes during the cold-start window
    where the router has nothing to differentiate.

    ``use_global_router`` (shared_A + ``route_per_layer=False``): the per-layer
    router is dropped; gates arrive via the broadcast ``_routing_weights``
    buffer from the network-level GlobalRouter. σ-band partition is rejected
    in this mode (no local logits to mask). Balance loss is silently inert —
    ``_last_gate`` is the detached broadcast.
    """

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
        num_experts=4,
        channel_scale=None,
        sigma_feature_dim: int = 0,
        expert_init_std: float = 0.0,
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: int = 1,
        sigma_bucket_boundaries: Optional[List[float]] = None,
        fei_feature_dim: int = 0,
        use_global_router: bool = False,
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

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.num_experts = num_experts
        self.in_dim = in_dim

        # Shared down projection.
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        # Per-expert up projections, fused: (E, out, r). Zero-init → ΔW=0 at
        # step 0. Symmetry is broken by `expert_warmup_ratio` (per-step
        # expert-gradient masking, LoRANetwork.step_expert_warmup).
        # `expert_init_std` is the paper-baseline knob (Tian et al. NeurIPS'24);
        # production runs leave it at 0.0.
        self.lora_up_weight = torch.nn.Parameter(
            torch.zeros(num_experts, out_dim, self.lora_dim)
        )
        if expert_init_std > 0.0:
            torch.nn.init.normal_(self.lora_up_weight, mean=0.0, std=expert_init_std)

        self.use_global_router = bool(use_global_router)
        # Local router on pooled rank-R (not raw in_dim): raw DiT inputs have
        # 80–96× DC-bias outliers + 4096 tokens, mean-pool collapses to DC and
        # the router gets no trainable gradient. lora_down is trained jointly,
        # so signal-carrying directions accumulate in rank-R space without
        # outlier saturation. See docs/methods/hydra-lora.md §Fixes.
        if self.use_global_router:
            self.sigma_feature_dim = 0
            self.fei_feature_dim = 0
        else:
            self.sigma_feature_dim = int(sigma_feature_dim)
            # FEI default (fei_dim=2) = raw 2-band simplex (e_low, e_high) from
            # library.runtime.fei.compute_fei_2band. See
            # ``[[project_fera_probe_2band_decision]]``.
            self.fei_feature_dim = int(fei_feature_dim)
            router_in_dim = (
                self.lora_dim + self.sigma_feature_dim + self.fei_feature_dim
            )
            self.router = torch.nn.Linear(router_in_dim, num_experts, bias=True)
            # Split init: small-std on rank-R columns, zeros on σ/FEI columns.
            # Step-0 gate matches σ=off+FEI=off; conditioning emerges as those
            # columns train.
            with torch.no_grad():
                self.router.weight.zero_()
                torch.nn.init.normal_(self.router.weight[:, : self.lora_dim], std=0.01)
                self.router.bias.zero_()

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        # Opt-in (shared down projection only): save bf16 x instead of fp32
        # x_lora for backward. Set by the network factory.
        self.use_custom_down_autograd = False

        self._last_gate = None  # (B, E), cached each forward for balance loss
        # σ / FEI / routing-weights placeholders: always-a-Tensor invariant +
        # pointer-stable buffers (see router_state.py). Routes through the
        # registration helpers so the cat / branch in ``_compute_gate`` runs
        # unconditionally — no None-vs-Tensor guard under compile_mode=full.
        _register_sigma_feature_cache(self, self.sigma_feature_dim)
        _register_fei_feature_cache(self, self.fei_feature_dim)
        if self.use_global_router:
            _register_routing_weights_buffer(self, num_experts)
        # σ-band partition: experts split into num_sigma_buckets bands;
        # out-of-band logits masked to -inf before softmax, soft routing
        # within each band. Independent of σ-feature router. Incompatible
        # with use_global_router — no local logits to mask.
        if specialize_experts_by_sigma_buckets and self.use_global_router:
            raise ValueError(
                "specialize_experts_by_sigma_buckets is incompatible with "
                "use_global_router=True (no per-layer logits to mask). Pick "
                "one: per-layer σ partition, or network-level GlobalRouter."
            )
        self._sigma_band_partition: bool = bool(specialize_experts_by_sigma_buckets)
        if self._sigma_band_partition:
            _register_sigma_band_partition(
                self, num_experts, num_sigma_buckets, sigma_bucket_boundaries
            )
        # Per-expert grad-scale mask (1.0 = full grad, 0.0 = stop-grad). All-
        # ones default makes ``up*1 + up.detach()*0`` collapse to ``up``, so
        # the forward branch is unconditional and dynamo doesn't recompile at
        # the warmup→post-warmup transition. step_expert_warmup mutates
        # in-place (dynamic, no recompile).
        self.register_buffer(
            "_expert_grad_mask",
            torch.ones(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def _compute_gate(self, lx: torch.Tensor) -> torch.Tensor:
        """RMS-pool rank-R signal, concat σ/FEI if enabled, router, softmax.

        RMS (not mean) pool: zero-mean activations don't cancel by √N over the
        L≈4096 sequence. Safe in rank-R space because lora_down strips the raw
        DiT 80–96× DC-bias outliers that break RMS in bf16 (see
        docs/methods/hydra-lora.md §Fixes).

        Under ``use_global_router`` ``lx`` is ignored — gate is the broadcast
        ``_routing_weights`` buffer.
        """
        if self.use_global_router:
            B = lx.shape[0] if lx.dim() >= 1 else 1
            w = self._routing_weights
            if w.dim() == 1:
                w = w.unsqueeze(0)
            return w.to(lx.dtype).expand(B, -1)
        if lx.dim() >= 3:
            B = lx.shape[0]
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        # lx is fp32 (bottleneck policy); router weights are in storage dtype.
        pooled = pooled.to(self.router.weight.dtype)
        parts = [pooled]
        if self.sigma_feature_dim > 0:
            # Placeholder (1, D) broadcasts to batch pre-set_sigma; expand is a
            # no-op once set_sigma rebinds to (B, D). Same rule for FEI below.
            sigma_feat = self._sigma_features.to(pooled.dtype).expand(
                pooled.shape[0], -1
            )
            parts.append(sigma_feat)
        if self.fei_feature_dim > 0:
            fei_feat = self._fei.to(pooled.dtype).expand(pooled.shape[0], -1)
            parts.append(fei_feat)
        router_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        logits = self.router(router_in)  # (B, E)
        if self._sigma_band_partition:
            logits = _apply_sigma_band_mask(
                logits, self._sigma, self._expert_band, self._sigma_edges
            )
        return torch.softmax(logits, dim=-1)

    def set_sigma(
        self, sigmas: torch.Tensor, sigma_features: torch.Tensor | None = None
    ) -> None:
        _set_sigma_feature_cache(self, sigmas, sigma_features)

    def clear_sigma(self) -> None:
        _clear_sigma_feature_cache(self)

    def set_fei(self, fei: torch.Tensor) -> None:
        _set_fei_feature_cache(self, fei)

    def clear_fei(self) -> None:
        _clear_fei_feature_cache(self)

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        # Shared helper preserves grad_fn — see router_state._set_routing_weights.
        if not getattr(self, "use_global_router", False):
            return
        _set_routing_weights(self, weights)

    def clear_routing_weights(self) -> None:
        if not getattr(self, "use_global_router", False):
            return
        _clear_routing_weights(self)

    def forward(self, x):
        # bf16 storage, fp32 bottleneck matmuls (see LoRAModule.forward).
        # Gate/router stays in autocast dtype — softmax over E is fine in bf16
        # given the small-std router init.
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        if self.use_custom_down_autograd and self.training:
            inv_scale = self.inv_scale if self._has_channel_scale else None
            lx = lora_down_project(x, self.lora_down.weight, inv_scale)
        else:
            x_lora = self._rebalance(x)
            lx = torch.nn.functional.linear(
                x_lora.float(), self.lora_down.weight.float()
            )

        # Gate from rank-R signal pre-mask/dropout — those are training-time
        # perturbations and the gate must behave identically at inference.
        gate = self._compute_gate(lx)  # (B, E)
        if self.training:
            # Plain STORE_ATTR (NOT @compiler.disable): a disabled helper
            # forces a graph break per LoRA forward and explodes
            # saved-for-backward memory under compile_mode=full (observed
            # OOM at 56 MoE + 140 OrthoLoRAExp modules on T4-class budget).
            self._last_gate = gate

        if self.training:
            lx = lx * self._timestep_mask

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # Expert-warmup masking: gradient flows only into the chosen expert's
        # slice during warmup, breaking the cold-start deadlock under a
        # near-uniform router. Outside warmup the mask is all-ones and
        # ``up*1 + up.detach()*0 == up`` (autograd-equivalent), so the branch
        # is unconditional — no Python-bool guard for dynamo to recompile on.
        up_weight = self.lora_up_weight
        expert_mask = self._expert_grad_mask.to(up_weight.dtype).view(-1, 1, 1)
        up_weight = up_weight * expert_mask + up_weight.detach() * (1.0 - expert_mask)

        # Gate-weighted up projection: (B, out, r) per batch element.
        combined = torch.einsum("be,eod->bod", gate.float(), up_weight.float())
        orig_shape = lx.shape
        B = orig_shape[0]
        lx_3d = lx.reshape(B, -1, orig_shape[-1])
        out = torch.bmm(lx_3d, combined.transpose(1, 2))
        out = out.reshape(*orig_shape[:-1], -1)

        return org_forwarded + (out * self.multiplier * scale).to(org_forwarded.dtype)
