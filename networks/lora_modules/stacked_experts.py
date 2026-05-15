# StackedExpertsLoRAModule: FeRA-style independent-A multi-LoRA experts.
#
# Each expert owns its own (lora_down, lora_up) — distinct from Hydra's
# shared-A layout. Independent-A is the defining trait of the FeRA paper
# (Yin et al., arXiv:2511.17979): experts specialize on disjoint sub-features
# rather than compete inside a shared pooled subspace.
#
# This module carries no router; gates arrive via `_routing_weights`. Owner
# is the network-level GlobalRouter (cfg.route_per_layer=False).
#
# Modes:
#   * Free (ortho=False): lora_down_weight (E, r, in) Kaiming, lora_up_weight
#     (E, out, r) zero-init.
#   * Ortho (ortho=True): PSOFT-style — frozen P_basis (out, r) + Q_basis
#     (r, in) from top-r SVD; per-expert S_q, S_p (E, r, r) Cayley-rotated;
#     per-expert lambda_layer (E, r) zero-init. Effective ΔW for expert e is
#     `P_basis @ cayley(S_p_e) @ diag(λ_e) @ cayley(S_q_e) @ Q_basis`. Symmetry
#     broken by small random S init (`ortho_init_std`) so the global router
#     has gradient signal from step 0.
#
# Activation-memory: stacked Parameters + two einsum boundaries save one
# (..., E, r) activation for backward instead of E × (..., out) from a
# per-expert ModuleList — ~50× less per-Linear autograd memory at (E, r)=(3, 8)
# on Anima MLP shapes.

import math

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.router_state import (
    _clear_routing_weights,
    _register_routing_weights_buffer,
    _set_routing_weights,
)


class StackedExpertsLoRAModule(BaseLoRAModule):
    """Independent-A multi-expert LoRA, gated from a broadcast buffer.

    Free mode params: lora_down_weight (E, r, in), lora_up_weight (E, out, r).
    Ortho mode params: S_q, S_p (E, r, r), lambda_layer (E, r); frozen
    P_basis (out, r) and Q_basis (r, in); _eye_r for batched Cayley solve.

    Both modes register a (1, E) `_routing_weights` placeholder rebound to
    (B, E) by `LoRANetwork.set_routing_weights`.

    T-LoRA composes via the inherited `_timestep_mask` (1, r), broadcast over
    the expert axis. `rank_dropout` is unsupported here — the base helper
    expects 2D/3D/4D lx; this forward produces 4D (B, L, E, r).
    """

    # Anima's adapted targets are projection Linears.
    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        num_experts: int = 3,
        channel_scale=None,
        ortho: bool = False,
        ortho_init_std: float = 0.02,
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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = int(num_experts)
        self.ortho = bool(ortho)
        self.ortho_init_std = float(ortho_init_std)

        if self.ortho:
            init_device = "cuda" if torch.cuda.is_available() else "cpu"
            W = org_module.weight.data.float().to(init_device)
            q = min(self.lora_dim + 6, min(W.shape))
            U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
            P_init = U[:, : self.lora_dim].clone().contiguous()
            Q_init = V[:, : self.lora_dim].T.clone().contiguous()
            del U, _S_vals, V, W

            self.register_buffer("P_basis", P_init.cpu())
            self.register_buffer("Q_basis", Q_init.cpu())

            # Random S init: zero-init with deterministic SVD basis would
            # leave every expert bit-identical (zero λ → zero gradient signal
            # for the router). ortho_init_std controls how far each expert
            # starts from identity rotation.
            self.S_p = torch.nn.Parameter(
                torch.randn(self.num_experts, self.lora_dim, self.lora_dim)
                * self.ortho_init_std
            )
            self.S_q = torch.nn.Parameter(
                torch.randn(self.num_experts, self.lora_dim, self.lora_dim)
                * self.ortho_init_std
            )

            # Zero-init λ → ΔW=0 at step 0 even though S is non-zero.
            self.lambda_layer = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.lora_dim)
            )

            self.register_buffer(
                "_eye_r",
                torch.eye(self.lora_dim, dtype=torch.float32),
                persistent=False,
            )

            # Channel rebalance absorbs into frozen Q_basis.
            if channel_scale is not None:
                self._register_channel_scale(self.Q_basis, channel_scale)
        else:
            # One stacked Parameter per side; expert axis leads. See file
            # header for the activation-memory rationale.
            self.lora_down_weight = torch.nn.Parameter(
                torch.empty(self.num_experts, self.lora_dim, in_dim)
            )
            self.lora_up_weight = torch.nn.Parameter(
                torch.zeros(self.num_experts, out_dim, self.lora_dim)
            )
            for k in range(self.num_experts):
                torch.nn.init.kaiming_uniform_(self.lora_down_weight[k], a=math.sqrt(5))

            # Same rebalance per slice — experts share input space. Repeat
            # _register_channel_scale calls are idempotent (same inv_scale).
            if channel_scale is not None:
                for k in range(self.num_experts):
                    self._register_channel_scale(
                        self.lora_down_weight[k], channel_scale
                    )

        _register_routing_weights_buffer(self, self.num_experts)

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        _set_routing_weights(self, weights)

    def clear_routing_weights(self) -> None:
        _clear_routing_weights(self)

    def _cayley_rotations(self):
        """Stacked S_q + S_p → one (2E, r, r) solve. Returns R_q, R_p (E, r, r)."""
        E = self.num_experts
        skew = torch.cat([self.S_q.float(), self.S_p.float()], dim=0)
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
        return R[:E], R[E:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded
        if self._skip_module():
            return org_forwarded

        # _routing_weights is (B, E); broadcast to (B, 1, ..., 1, E, 1) over
        # the (B, ..., E, r) rank-level activations.
        w = self._routing_weights

        if self.ortho:
            compute_dtype = self.P_basis.dtype
            x_lora = self._rebalance(x.to(compute_dtype))

            R_q, R_p = self._cayley_rotations()
            R_q = R_q.to(compute_dtype)
            R_p = R_p.to(compute_dtype)

            # Shared down boundary, then per-expert R_q rotation.
            x_proj = torch.nn.functional.linear(x_lora, self.Q_basis)
            lx = torch.einsum("...j,eij->...ei", x_proj, R_q)

            lx = lx * self.lambda_layer.to(compute_dtype)
            # _timestep_mask (1, r) broadcasts uniformly over the expert axis.
            lx = lx * self._timestep_mask

            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            B = w.shape[0]
            n_mid = lx.ndim - 3
            view_shape = (B,) + (1,) * n_mid + (self.num_experts, 1)
            lx = lx * w.view(view_shape).to(compute_dtype)

            # Per-expert R_p + sum-over-experts in one einsum, then shared P.
            mid = torch.einsum("ejr,...er->...j", R_p, lx)
            adapter = torch.nn.functional.linear(mid, self.P_basis)
        else:
            # bf16 storage, fp32 bottleneck — matches Hydra free-mode.
            x_lora = self._rebalance(x)

            # Batched down: (..., in) @ (E, r, in)^T → (..., E, r). Saves ONE
            # (..., E, r) activation vs E × (..., out) from a per-expert loop.
            lx = torch.einsum(
                "...i,eri->...er",
                x_lora.float(),
                self.lora_down_weight.float(),
            )

            lx = lx * self._timestep_mask
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            B = w.shape[0]
            n_mid = lx.ndim - 3
            view_shape = (B,) + (1,) * n_mid + (self.num_experts, 1)
            lx = lx * w.view(view_shape).float()

            # Batched up: (..., E, r) @ (E, out, r)^T → (..., out).
            adapter = torch.einsum("...er,eor->...o", lx, self.lora_up_weight.float())

        lora_out = adapter * self.multiplier * self.scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley structural in ortho mode, no constraint in free mode."""
        device = self.S_p.device if self.ortho else self.lora_down_weight.device
        zero = torch.tensor(0.0, device=device)
        return zero, zero
