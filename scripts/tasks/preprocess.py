"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ._common import PY, _load_subset_configs, _path, run


# Subfolders under the source dir are walked by default — matches the
# `recursive = true` subset default in configs/base.toml. Stems must stay
# unique across the tree (cache filenames are stem-keyed and flat). Pass
# `--no_recursive` (or edit configs) to opt out.
def cmd_preprocess_resize(extra):
    run(
        [
            PY,
            "preprocess/resize_images.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            "--dst",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--no_copy_captions",
            "--recursive",
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
            "--recursive",
            *extra,
        ]
    )


def cmd_preprocess_te(extra):
    # CAPTION_SHUFFLE_VARIANTS / CAPTION_TAG_DROPOUT_RATE let the GUI's
    # Preprocessing tab control these without editing this file. Defaults
    # match the historical hardcoded values so non-GUI invocations are
    # unchanged.
    shuffle_variants = os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
    tag_dropout_rate = os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            _path("source_image_dir", "image_dataset"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
            "--recursive",
            *extra,
        ]
    )


def cmd_preprocess_pooled(extra):
    """Cache pooled text embeddings (max over seq dim) from existing TE caches.

    Reads ``{stem}_anima_te.safetensors`` from the LoRA cache dir and writes
    ``{stem}_anima_pooled.safetensors`` sidecars next to them. Consumed by
    ``make distill-mod`` to skip a redundant ``.max(dim=1)`` per training
    microstep / val sigma. No GPU needed.
    """
    run(
        [
            PY,
            "preprocess/cache_pooled_text.py",
            "--dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            *extra,
        ]
    )


def cmd_preprocess_pe(extra):
    """Cache PE-Core-L14-336 vision-encoder features.

    Reads pre-resized images from ``post_image_dataset/resized/`` (the
    standard LoRA pipeline source) and writes
    ``{stem}_anima_pe.safetensors`` sidecars into the LoRA cache dir so the
    dataset's existing ``cache_dir`` lookup finds them.

    Consumed by methods that align against frozen vision features —
    currently REPA (--use_repa) and IP-Adapter when reading PE features off
    disk.
    """
    run(
        [
            PY,
            "preprocess/cache_pe_encoder.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--encoder",
            "pe",
            "--recursive",
            *extra,
        ]
    )


def _common_parent(paths: list[str]) -> str | None:
    if not paths:
        return None
    resolved = [Path(p).resolve() for p in paths]
    try:
        return str(os.path.commonpath(resolved))
    except ValueError:
        return None


def _derive_tree_paths(subsets: list[dict]) -> tuple[str, str, str] | None:
    sources, images, caches = [], [], []
    for s in subsets:
        sd = s.get("source_dir")
        id_ = s.get("image_dir")
        cd = s.get("cache_dir")
        if not sd or not id_ or not cd:
            return None
        sources.append(sd)
        images.append(id_)
        caches.append(cd)
    source_parent = _common_parent(sources)
    if not source_parent:
        return None
    image_parents = [str(Path(p).parent) for p in images]
    dst = _common_parent(image_parents)
    if not dst:
        return None
    cache_parents = [str(Path(p).parent) for p in caches]
    cache_base = _common_parent(cache_parents)
    if not cache_base:
        return None
    if dst != cache_base:
        return None
    return source_parent, dst, cache_base


def _cmd_preprocess_subsets_tree(subsets, source_dir, dst, cache_dir, extra):
    print(
        f"  cmd_preprocess_subsets: tree mode — "
        f"source_dir={source_dir!r}  dst={dst!r}  cache_dir={cache_dir!r}",
        file=sys.stderr,
    )
    for i, subset in enumerate(subsets):
        name = subset.get("name", "")
        sd = subset.get("source_dir", "?")
        id_ = subset.get("image_dir", "?")
        cd = subset.get("cache_dir", "?")
        print(f"  subset[{i}] ({name!r}): src={sd!r} → dst={id_!r}  cache={cd!r}", file=sys.stderr)
    src = Path(source_dir)
    if not src.is_dir():
        print(f"  cmd_preprocess_subsets: source_dir {source_dir!r} does not exist", file=sys.stderr)
        return
    run(
        [
            PY,
            "preprocess/resize_images.py",
            "--src",
            source_dir,
            "--dst",
            dst,
            "--tree",
            "--no_copy_captions",
            *extra,
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            dst,
            "--cache_dir",
            dst,
            "--tree",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
            *extra,
        ]
    )
    shuffle_variants = os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
    tag_dropout_rate = os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            source_dir,
            "--cache_dir",
            dst,
            "--tree",
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
            *extra,
        ]
    )


def cmd_preprocess_subsets(extra):
    subsets = _load_subset_configs()
    if not subsets:
        print("  cmd_preprocess_subsets: no subset configs found, nothing to do", file=sys.stderr)
        return
    print(f"  cmd_preprocess_subsets: processing {len(subsets)} subset(s)...", file=sys.stderr)
    tree_paths = _derive_tree_paths(subsets)
    if tree_paths is not None:
        source_dir, dst, cache_dir = tree_paths
        _cmd_preprocess_subsets_tree(subsets, source_dir, dst, cache_dir, extra)
        print(f"  cmd_preprocess_subsets: all {len(subsets)} subset(s) processed (tree mode)", file=sys.stderr)
        return
    print("  cmd_preprocess_subsets: could not derive common tree paths, falling back to per-subset mode", file=sys.stderr)
    for i, subset in enumerate(subsets):
        source_dir = subset.get("source_dir")
        image_dir = subset.get("image_dir")
        cache_dir = subset.get("cache_dir")
        name = subset.get("name", "")
        if not source_dir or not image_dir or not cache_dir:
            print(
                f"  subset[{i}] ({name!r}): missing path keys — "
                f"source_dir={source_dir!r}  image_dir={image_dir!r}  cache_dir={cache_dir!r}, skipping",
                file=sys.stderr,
            )
            continue
        src = Path(source_dir)
        if not src.is_dir():
            print(f"  subset[{i}] ({name!r}): source_dir {source_dir!r} does not exist, skipping", file=sys.stderr)
            continue
        is_root = name == "(root)"
        recursive_flag = "--no-recursive (root subset)" if is_root else "--recursive"
        print(
            f"  subset[{i}] ({name!r}): src={source_dir!r} → dst={image_dir!r}  cache={cache_dir!r}  {recursive_flag}",
            file=sys.stderr,
        )
        resize_cmd = [
            PY,
            "preprocess/resize_images.py",
            "--src",
            source_dir,
            "--dst",
            image_dir,
            "--no_copy_captions",
        ]
        if not is_root:
            resize_cmd.append("--recursive")
        resize_cmd.extend(extra)
        run(resize_cmd)
        run(
            [
                PY,
                "preprocess/cache_latents.py",
                "--dir",
                image_dir,
                "--cache_dir",
                cache_dir,
                "--vae",
                "models/vae/qwen_image_vae.safetensors",
                "--batch_size",
                "4",
                "--chunk_size",
                "64",
                "--recursive",
                *extra,
            ]
        )
        shuffle_variants = os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
        tag_dropout_rate = os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
        te_cmd = [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            source_dir,
            "--cache_dir",
            cache_dir,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
        ]
        if not is_root:
            te_cmd.append("--recursive")
        te_cmd.extend(extra)
        run(te_cmd)
    print(f"  cmd_preprocess_subsets: all {len(subsets)} subset(s) processed", file=sys.stderr)


def cmd_preprocess(extra):
    # PE features are intentionally NOT cached here — only REPA / IP-Adapter /
    # CMMD need them, and those paths chain `preprocess-pe` explicitly (see
    # `exp-ip-adapter-preprocess`). Leaving PE out keeps the default LoRA
    # preprocess fast on machines that won't ever use the vision tower.

    subsets = _load_subset_configs()
    if subsets:
        print(f"  cmd_preprocess: multi-subset mode detected ({len(subsets)} subset(s))", file=sys.stderr)
        cmd_preprocess_subsets(extra)
        return
    print("  cmd_preprocess: single-dataset mode (no subset configs found)", file=sys.stderr)
    cmd_preprocess_resize(extra)
    cmd_preprocess_vae(extra)
    cmd_preprocess_te(extra)
