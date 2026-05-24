"""Save-pipeline orchestrator for the LoRA / Ortho / Hydra family.

The per-variant save logic — Cayley distillation, MoE write layout,
qkv defuse — lives on the variant's module class in
``networks/lora_modules/`` (``OrthoLoRAModule.distill_save_state_dict``,
``HydraLoRAModule.build_moe_state_dict``, etc). This file is the thin
ordering layer that calls them and writes the resulting file(s).

Ordering of the conversion pipeline is load-bearing:

  1. ``ChimeraHydraLoRAModule.distill_save_state_dict``
     (gated on co-located ``.S_q_c`` + ``.S_q_f``)
  2. ``StackedExpertsLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 3-D ``.S_q``)
  3. ``OrthoHydraLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 2-D ``.S_q``)
  4. ``OrthoLoRAModule.distill_save_state_dict``
     (gated on 2-D ``.S_p``)
  5. legacy sig-type OrthoLoRA → standard LoRA
     (gated on ``.base_lambda``; kept here because it touches the
     deprecated ``lora_deprecated.OrthoLoRAModule`` save layout that no
     live module class owns)

The ``.S_p`` / ``.S_q`` dimensionality is the discriminator — every step
checks both dims explicitly so the matchers never overlap on the same
prefix. Step 5 handles legacy checkpoints from
``lora_deprecated.OrthoLoRAModule``; current training never emits those
keys, but the converter is kept so old artifacts remain re-bakeable.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from library.log import setup_logging
from networks.lora_modules import (
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
)
from networks.lora_modules.lora import defuse_and_bake_standard
from networks.attn_fuse import match_fused_spec

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy: sig-type OrthoLoRA → standard LoRA via 2r-dim SVD.
#
# Kept here (not on a module class) because the live ``OrthoLoRAModule``
# never emits these keys — they belong to the deprecated
# ``lora_deprecated.OrthoLoRAModule``, which is gone from the runtime path.
# ---------------------------------------------------------------------------


def _convert_legacy_ortho_to_lora(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".base_lambda"):
            prefixes.add(key[: -len(".base_lambda")])

    for prefix in prefixes:
        P = state_dict[f"{prefix}.p_layer.weight"]  # (out, r)
        Q = state_dict[f"{prefix}.q_layer.weight"]  # (r, in)
        lam = state_dict[f"{prefix}.lambda_layer"]
        P_base = state_dict[f"{prefix}.base_p_weight"]
        Q_base = state_dict[f"{prefix}.base_q_weight"]
        lam_base = state_dict[f"{prefix}.base_lambda"]
        alpha = state_dict.get(f"{prefix}.alpha")
        rank = Q.shape[0]

        # ΔW = P·diag(λ)·Q − P_base·diag(λ_base)·Q_base is rank ≤ 2r. SVD
        # works in the small 2r-dim column/row space instead of on the full
        # (out × in) matrix: ΔW = [P|P_base] @ M @ [Q; Q_base], then SVD of M.
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        save_dtype = dtype if dtype is not None else P.dtype

        P_cat = torch.cat([P, P_base], dim=1).float().to(svd_device)  # (out, 2r)
        Q_cat = torch.cat([Q, Q_base], dim=0).float().to(svd_device)  # (2r, in)
        lam_diag = torch.diag(lam.squeeze(0).float().to(svd_device))
        lam_base_diag = torch.diag(lam_base.squeeze(0).float().to(svd_device))

        M = torch.zeros(2 * rank, 2 * rank, device=svd_device)
        M[:rank, :rank] = lam_diag
        M[rank:, rank:] = -lam_base_diag

        Qp, Rp = torch.linalg.qr(P_cat)
        Qq, Rq = torch.linalg.qr(Q_cat.T)

        core = Rp @ M @ Rq.T
        Uc, Sc, Vhc = torch.linalg.svd(core)

        lora_up = (
            (Qp @ Uc[:, :rank] * Sc[:rank].sqrt().unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )
        lora_down = (
            (Sc[:rank].sqrt().unsqueeze(1) * Vhc[:rank, :] @ Qq.T)
            .to(save_dtype)
            .cpu()
            .contiguous()
        )

        for suffix in (
            "p_layer.weight",
            "q_layer.weight",
            "lambda_layer",
            "base_p_weight",
            "base_q_weight",
            "base_lambda",
        ):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


# ---------------------------------------------------------------------------
# Back-compat shim: tests/test_global_router.py imports this name directly
# to exercise the StackedExperts MoE writer in isolation.
# ---------------------------------------------------------------------------


def _build_stacked_experts_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Thin shim → :meth:`StackedExpertsLoRAModule.build_moe_state_dict`."""
    return StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def defuse_loha_qkv(state_dict):
    fused_groups = []
    for key in list(state_dict.keys()):
        if not key.endswith(".hada_w1_b"):
            continue
        prefix = key.removesuffix(".hada_w1_b")
        spec = match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        component_letters = spec.component_letters
        n = len(component_letters)
        is_tucker = f"{prefix}.hada_t1" in state_dict

        w1_a = state_dict.pop(f"{prefix}.hada_w1_a", None)
        w1_b = state_dict.pop(f"{prefix}.hada_w1_b")
        w2_a = state_dict.pop(f"{prefix}.hada_w2_a", None)
        w2_b = state_dict.pop(f"{prefix}.hada_w2_b")
        alpha = state_dict.pop(f"{prefix}.alpha", None)
        t1 = state_dict.pop(f"{prefix}.hada_t1", None) if is_tucker else None
        t2 = state_dict.pop(f"{prefix}.hada_t2", None) if is_tucker else None

        if is_tucker:
            if w1_a.shape[1] % n != 0 or w2_a.shape[1] % n != 0:
                raise ValueError(
                    f"defuse_loha_qkv: Tucker w1_a dim1={w1_a.shape[1]} or w2_a dim1={w2_a.shape[1]} "
                    f"not divisible by n={n} for prefix={prefix}"
                )
            w1_a_chunks = w1_a.chunk(n, dim=1)
            w2_a_chunks = w2_a.chunk(n, dim=1)
        else:
            if w1_a.shape[0] % n != 0 or w2_a.shape[0] % n != 0:
                raise ValueError(
                    f"defuse_loha_qkv: w1_a dim0={w1_a.shape[0]} or w2_a dim0={w2_a.shape[0]} "
                    f"not divisible by n={n} for prefix={prefix}"
                )
            w1_a_chunks = w1_a.chunk(n, dim=0)
            w2_a_chunks = w2_a.chunk(n, dim=0)

        base_prefix = prefix.removesuffix(spec.fused_frag)
        for letter, w1_a_c, w2_a_c in zip(component_letters, w1_a_chunks, w2_a_chunks):
            new_prefix = base_prefix + spec.component_frag(letter)
            state_dict[f"{new_prefix}.hada_w1_a"] = w1_a_c
            state_dict[f"{new_prefix}.hada_w1_b"] = w1_b.clone()
            state_dict[f"{new_prefix}.hada_w2_a"] = w2_a_c
            state_dict[f"{new_prefix}.hada_w2_b"] = w2_b.clone()
            if alpha is not None:
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()

            if is_tucker:
                state_dict[f"{new_prefix}.hada_t1"] = t1.clone()
                state_dict[f"{new_prefix}.hada_t2"] = t2.clone()


def defuse_lokr_qkv(state_dict):
    fused_groups = []
    for key in list(state_dict.keys()):
        for sentinel in (".lokr_w1", ".lokr_w1_b"):
            if key.endswith(sentinel):
                prefix = key.removesuffix(sentinel)
                spec = match_fused_spec(prefix)
                if spec is not None:
                    fused_groups.append((prefix, spec))
                break

    seen = set()
    deduped_groups = []
    for prefix, spec in fused_groups:
        if prefix not in seen:
            seen.add(prefix)
            deduped_groups.append((prefix, spec))

    for prefix, spec in deduped_groups:
        component_letters = spec.component_letters
        n = len(component_letters)

        use_w1 = f"{prefix}.lokr_w1" in state_dict
        use_w2 = f"{prefix}.lokr_w2" in state_dict

        if use_w1:
            split_dim = state_dict[f"{prefix}.lokr_w1"].shape[0]
        else:
            split_dim = state_dict[f"{prefix}.lokr_w1_a"].shape[0]
        if split_dim < n or split_dim % n != 0:
            logger.warning(
                "defuse_lokr_qkv: %s lokr_w1 split_dim=%d not divisible by n=%d, "
                "falling back to full delta split (may produce large checkpoint)",
                prefix,
                split_dim,
                n,
            )
            w1_key = f"{prefix}.lokr_w1" if use_w1 else f"{prefix}.lokr_w1_a"
            w1b_key = f"{prefix}.lokr_w1_b" if not use_w1 else None
            w2_key = f"{prefix}.lokr_w2" if use_w2 else None
            w2a_key = f"{prefix}.lokr_w2_a" if not use_w2 else None
            w2b_key = f"{prefix}.lokr_w2_b" if not use_w2 else None
            t2_key = f"{prefix}.lokr_t2"
            alpha_key = f"{prefix}.alpha"
            dora_key = f"{prefix}.dora_scale"
            has_t2 = t2_key in state_dict
            has_dora = dora_key in state_dict

            alpha_val = state_dict.pop(alpha_key, None)
            dora_val = state_dict.pop(dora_key, None) if has_dora else None
            lora_dim = state_dict[w1b_key].shape[0] if w1b_key else 1

            if use_w1:
                w1_val = state_dict.pop(w1_key)
            else:
                w1a_val = state_dict.pop(w1_key)
                w1b_val = state_dict.pop(w1b_key)
                w1_val = w1a_val @ w1b_val
            if use_w2:
                w2_val = state_dict.pop(w2_key)
            else:
                w2a_val = state_dict.pop(w2a_key)
                w2b_val = state_dict.pop(w2b_key)
                t2_val = state_dict.pop(t2_key, None) if has_t2 else None
                if t2_val is not None:
                    from networks.lora_modules.lycoris_functional import rebuild_tucker

                    w2_val = rebuild_tucker(t2_val, w2a_val, w2b_val)
                else:
                    w2_val = w2a_val @ w2b_val

            from networks.lora_modules.lycoris_functional import make_kron

            scale = (
                1.0
                if (use_w1 and use_w2)
                else (
                    float(alpha_val.item()) / lora_dim if alpha_val is not None else 1.0
                )
            )
            delta = make_kron(w1_val.float(), w2_val.float(), scale)
            if delta.shape[0] % n != 0:
                state_dict[w1_key] = w1_val
                if alpha_val is not None:
                    state_dict[alpha_key] = alpha_val
                if dora_val is not None:
                    state_dict[dora_key] = dora_val
                continue
            delta_chunks = delta.chunk(n, dim=0)
            dora_chunks = dora_val.chunk(n, dim=0) if dora_val is not None else None

            base_prefix = prefix.removesuffix(spec.fused_frag)
            for i, letter in enumerate(component_letters):
                new_prefix = base_prefix + spec.component_frag(letter)
                chunk_i = delta_chunks[i].clone()
                per_out = chunk_i.shape[0]
                per_in = (
                    chunk_i.shape[1]
                    if chunk_i.dim() == 2
                    else chunk_i.shape[1]
                    * int(torch.tensor(chunk_i.shape[2:]).prod().item())
                )
                state_dict[f"{new_prefix}.lokr_w1"] = (
                    chunk_i.reshape(per_out, per_in) if chunk_i.dim() > 2 else chunk_i
                )
                state_dict[f"{new_prefix}.lokr_w2"] = (
                    torch.ones(1, 1, *chunk_i.shape[2:])
                    if chunk_i.dim() > 2
                    else torch.ones(1, 1)
                )
                state_dict[f"{new_prefix}.alpha"] = torch.tensor(float(lora_dim))
                if dora_chunks is not None:
                    state_dict[f"{new_prefix}.dora_scale"] = dora_chunks[i].clone()
            for k in [w1_key, w1b_key, w2_key, w2a_key, w2b_key, t2_key]:
                if k is not None:
                    state_dict.pop(k, None)
            continue

        alpha = state_dict.pop(f"{prefix}.alpha", None)
        dora_scale = state_dict.pop(f"{prefix}.dora_scale", None)
        dora_scale_chunks = (
            dora_scale.chunk(n, dim=0) if dora_scale is not None else None
        )

        if use_w1:
            w1 = state_dict.pop(f"{prefix}.lokr_w1")
            w1_chunks = w1.chunk(n, dim=0)
        else:
            w1_a = state_dict.pop(f"{prefix}.lokr_w1_a")
            w1_b = state_dict.pop(f"{prefix}.lokr_w1_b")
            w1_a_chunks = w1_a.chunk(n, dim=0)

        if use_w2:
            w2 = state_dict.pop(f"{prefix}.lokr_w2")
        else:
            w2_a = state_dict.pop(f"{prefix}.lokr_w2_a")
            w2_b = state_dict.pop(f"{prefix}.lokr_w2_b")
            t2 = state_dict.pop(f"{prefix}.lokr_t2", None)

        base_prefix = prefix.removesuffix(spec.fused_frag)
        for i, letter in enumerate(component_letters):
            new_prefix = base_prefix + spec.component_frag(letter)

            if use_w1:
                state_dict[f"{new_prefix}.lokr_w1"] = w1_chunks[i]
            else:
                state_dict[f"{new_prefix}.lokr_w1_a"] = w1_a_chunks[i]
                state_dict[f"{new_prefix}.lokr_w1_b"] = w1_b.clone()

            if use_w2:
                state_dict[f"{new_prefix}.lokr_w2"] = w2.clone()
            else:
                state_dict[f"{new_prefix}.lokr_w2_a"] = w2_a.clone()
                state_dict[f"{new_prefix}.lokr_w2_b"] = w2_b.clone()
                if t2 is not None:
                    state_dict[f"{new_prefix}.lokr_t2"] = t2.clone()

            if alpha is not None:
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()

            if dora_scale_chunks is not None:
                state_dict[f"{new_prefix}.dora_scale"] = dora_scale_chunks[i]


def save_network_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    save_variant: str,
) -> None:
    """Run the full save pipeline: distill chain + variant write.

    Mutates ``state_dict`` in place.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    # Distill chain. Order is load-bearing — see module docstring.
    ChimeraHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    StackedExpertsLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoLoRAModule.distill_save_state_dict(state_dict, dtype)
    _convert_legacy_ortho_to_lora(state_dict, dtype)

    # Variant dispatch.
    #   * ``stacked_experts_global_fei``: independent-A per-expert
    #     ``(lora_downs.{i}, lora_ups.{i})`` → ``*_moe.safetensors``.
    #   * ``chimera_hydra_moe``: dual-A per-pool ``lora_{down,up}_{c,f}`` +
    #     ``freq_router.*`` → ``*_chimera.safetensors``.
    #   * ``hydra_moe`` / ``ortho_hydra_to_hydra``: shared-A Hydra
    #     ``(lora_down, lora_ups.{i})`` → ``*_moe.safetensors``.
    #   * standard: defuse qkv → ``*.safetensors``.
    #
    # Auto-fallback for hydra: any ``.lora_up_weight`` key surviving the
    # distill chain implies a Hydra payload. Kept for callers that don't
    # plumb ``save_variant`` through.
    is_stacked_experts_variant = save_variant == "stacked_experts_global_fei"
    is_chimera_variant = save_variant == "chimera_hydra_moe"
    is_hydra_variant = (
        save_variant in ("hydra_moe", "ortho_hydra_to_hydra")
        or (
            not is_chimera_variant
            and any(k.endswith(".lora_up_weight") for k in state_dict.keys())
        )
    ) and not is_stacked_experts_variant

    if is_stacked_experts_variant:
        se_file = os.path.splitext(file)[0] + "_moe.safetensors"
        se_sd = StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(se_sd, se_file, metadata or {})
        logger.info(f"StackedExperts full format saved to {se_file}")
        return

    if is_chimera_variant:
        chimera_file = os.path.splitext(file)[0] + "_chimera.safetensors"
        chimera_sd = ChimeraHydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(chimera_sd, chimera_file, metadata or {})
        logger.info(f"ChimeraHydra full format saved to {chimera_file}")
        return

    if is_hydra_variant:
        hydra_file = os.path.splitext(file)[0] + "_moe.safetensors"
        hydra_sd = HydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(hydra_sd, hydra_file, metadata or {})
        logger.info(f"HydraLoRA full format saved to {hydra_file}")
        # The _moe file is the only useful artifact for HydraLoRA —
        # a uniform expert average defeats layer-local routing.
        return

    # Standard (lora / ortho / dora) write path.
    defuse_loha_qkv(state_dict)
    defuse_lokr_qkv(state_dict)
    defuse_and_bake_standard(state_dict)

    if dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key].detach().clone().to("cpu").to(dtype)
            state_dict[key] = v

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)
