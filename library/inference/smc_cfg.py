"""Sliding-Mode Control CFG (SMC-CFG).

Wang et al., "CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance",
arXiv:2603.03281. Algorithm 1, with classical SMC boundary-layer smoothing.

Drop-in modification of the CFG cond/uncond combine. At each denoising step:

    e_t        = v_cond − v_uncond            (semantic error, velocity-space)
    s_t        = (e_t − e_prev) + λ · e_prev  (sliding-mode surface)
    Δe         = −k · φ(s_t / ε)              (switching correction)
    v̂_t        = v_uncond + w · (e_t + Δe)

φ(·) is the switching function. Paper uses sign(·); we default to tanh(·)
with an auto-derived boundary layer ε = mean(|s|) per step. tanh equals
sign in the limit ε→0 but smooths the per-element bang-bang behavior:
elements within roughly ±ε of zero get proportional treatment, outside
saturate near ±1. This is the textbook chattering reduction from
classical SMC (Edwards & Spurgeon, 1998) — the paper omits it, but
without it the per-element ±k injection is structurally noise-shaped at
small CFG / small |e| regimes (composition drift + visible texture noise
on Anima at CFG=4). Pass eps=0 to recover the paper's exact sign().

`e_prev` is the raw e from the previous step (None → e_prev := e_t on the
first step, matching the paper's `if e(t+1) is None then e(t+1) ← e(t)`).
We store the *uncontrolled* e_prev so the sliding surface tracks the real
discrepancy, not the controller's own feedback.

No extra DiT forwards. One velocity-shaped buffer of state.
Composes with DCW (post-step x-space correction) and mod-guidance (AdaLN-side).
"""

from __future__ import annotations

from typing import Optional

import torch


class SMCCFGState:
    def __init__(
        self,
        lam: float = 5.0,
        k: float = 0.1,
        eps: Optional[float] = None,
        alpha: Optional[float] = None,
    ):
        self.lam = float(lam)
        self.k = float(k)
        # eps: boundary layer thickness. None → auto (mean(|s|) per step).
        # 0.0 → paper-exact sign(s). >0 → fixed boundary layer.
        self.eps = None if eps is None else float(eps)
        # alpha: dimensionless adaptive gain. None → use fixed self.k.
        # Set → k_t = alpha · |e_t|.mean() per step (self-scales across
        # model / CFG / σ / sample; see bench/smc_cfg/analysis_and_proposal.md).
        # When set, self.k is ignored.
        self.alpha = None if alpha is None else float(alpha)
        self._e_prev: Optional[torch.Tensor] = None

    def combine(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        e = cond - uncond
        e_prev = e if self._e_prev is None else self._e_prev
        s = (e - e_prev) + self.lam * e_prev

        if self.eps == 0.0:
            switch = torch.sign(s)  # paper-exact, chattering
        else:
            if self.eps is None:
                eps_t = s.abs().mean().clamp_min(1e-8)
            else:
                eps_t = self.eps
            switch = torch.tanh(s / eps_t)

        # k_t: adaptive (α-scaled |e_t|.mean()) if alpha is set, else fixed self.k.
        # Adaptive form keeps the controller in-band across the trajectory by
        # construction — see bench/smc_cfg/analysis_and_proposal.md §A.
        if self.alpha is not None:
            k_t = self.alpha * e.abs().mean().clamp_min(1e-12)
        else:
            k_t = self.k

        delta_e = -k_t * switch
        self._e_prev = e.detach()
        return uncond + guidance_scale * (e + delta_e)
