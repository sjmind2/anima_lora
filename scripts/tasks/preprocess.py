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
def _min_pixels_args() -> list[str]:
    """``--min_pixels <N>`` derived from the variant TOML's
    ``drop_lowres_images`` + ``min_pixels`` keys (resolved through the same
    base → preset → method merge chain training uses, via ``_path_overrides``
    in scripts/tasks/_common.py).

    Returns ``[]`` when both keys are absent so plain CLI use keeps each
    script's own argparse default (500_000 = 0.5MP). ``drop_lowres_images
    = false`` forces ``--min_pixels 0`` even when ``min_pixels`` is set, so
    the user can flip a single boolean to disable the filter."""
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    if "drop_lowres_images" not in overrides and "min_pixels" not in overrides:
        return []
    if overrides.get("drop_lowres_images") is False:
        return ["--min_pixels", "0"]
    raw = overrides.get("min_pixels", 500_000)
    try:
        n = max(0, int(raw))
    except (TypeError, ValueError):
        return []
    return ["--min_pixels", str(n)]


def _config_min_pixels() -> int:
    """The configured ``min_pixels`` threshold (merged chain), default 0.5MP."""
    from ._common import _path_overrides

    raw = _path_overrides().get("min_pixels", 500_000)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 500_000


def _resolve_lowres_filter(extra) -> tuple[list[str], list[str]]:
    """Reconcile the low-res input filter against CLI ``ARGS``.

    Returns ``(min_pixels_args, cleaned_extra)`` where ``cleaned_extra`` has
    our two convenience flags popped so the underlying scripts never see an
    arg their argparse doesn't define. Precedence (highest first):

      1. An explicit ``--min_pixels N`` in ``ARGS`` — left in ``extra`` and
         wins outright; we inject nothing (no duplicate ``--min_pixels``).
      2. ``--no_drop_lowres`` in ``ARGS`` → ``--min_pixels 0`` (keep every
         image), overriding ``drop_lowres_images = true`` in the TOML.
      3. ``--drop_lowres`` in ``ARGS`` → force the configured ``min_pixels``
         threshold, overriding ``drop_lowres_images = false`` in the TOML.
      4. Neither flag → fall back to the merged-config behavior
         (``_min_pixels_args``)."""
    cleaned = list(extra)
    no_drop = "--no_drop_lowres" in cleaned
    drop = "--drop_lowres" in cleaned
    cleaned = [a for a in cleaned if a not in ("--no_drop_lowres", "--drop_lowres")]

    # An explicit threshold in ARGS is authoritative; leave it in place.
    if "--min_pixels" in cleaned:
        return [], cleaned
    if no_drop:  # disable wins over enable when both are passed
        return ["--min_pixels", "0"], cleaned
    if drop:
        return ["--min_pixels", str(_config_min_pixels())], cleaned
    return _min_pixels_args(), cleaned


def cmd_preprocess_resize(extra):
    mp_args, extra = _resolve_lowres_filter(extra)
    run(
        [
            PY,
            "scripts/preprocess/resize_images.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            "--dst",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--no_copy_captions",
            "--recursive",
            *mp_args,
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
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
    mp_args, extra = _resolve_lowres_filter(extra)
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
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
            *mp_args,
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
            "scripts/preprocess/cache_pooled_text.py",
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

    Consumed by IP-Adapter when reading PE features off disk and by the
    DCW v4 fusion head's pooled-image-feature input channel.
    """
    run(
        [
            PY,
            "scripts/preprocess/cache_pe_encoder.py",
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
            "scripts/preprocess/resize_images.py",
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
            "scripts/preprocess/cache_latents.py",
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
            "scripts/preprocess/cache_text_embeddings.py",
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


def cmd_caption_index(extra):
    """Build the method-agnostic typed-tag caption index.

    Walks caption sidecars under the source dir, classifies tags into
    character / copyright / artist / count via the Anima Tagger vocab, and
    writes ``post_image_dataset/captions/caption_index.json`` (per-image typed
    tags + group inversions). Pure data, no GPU. Consumed by the IP-Adapter
    distinct-pair sampler, artist balancing, and dataset analytics. Regenerate
    when the dataset or vocab changes.
    """
    run(
        [
            PY,
            "scripts/preprocess/build_caption_index.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
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
            "scripts/preprocess/resize_images.py",
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
                "scripts/preprocess/cache_latents.py",
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
            "scripts/preprocess/cache_text_embeddings.py",
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
    # PE features are intentionally NOT cached here — only IP-Adapter / CMMD /
    # DCW v4 need them, and those paths chain `preprocess-pe` explicitly (see
    # `exp-ip-adapter-preprocess`). Leaving PE out keeps the default LoRA
    # preprocess fast on machines that won't ever use the vision tower.

    subsets = _load_subset_configs()
    if subsets:
        print(f"  cmd_preprocess: multi-subset mode detected ({len(subsets)} subset(s))", file=sys.stderr)
        cmd_preprocess_subsets(extra)
        return
    print("  cmd_preprocess: single-dataset mode (no subset configs found)", file=sys.stderr)
    cmd_preprocess_resize(extra)
    # The VAE step doesn't filter on size; strip the low-res convenience flags
    # so its argparse never sees an arg it doesn't define. (resize/te pop them
    # themselves via _resolve_lowres_filter.)
    _, vae_extra = _resolve_lowres_filter(extra)
    cmd_preprocess_vae(vae_extra)
    cmd_preprocess_te(extra)


def cmd_preprocess_config(extra):
    """Preprocess the exact directories named in a ``--dataset_config`` TOML.

    Unlike ``cmd_preprocess`` (which resolves the repo's standard
    ``image_dataset/`` → ``post_image_dataset/`` layout from the merged
    config), this drives off the same dataset config the *training* job will
    consume, so one file fully describes an ad-hoc job — no reliance on the
    default layout. For each ``[[datasets.subsets]]`` it:

      1. bucket-resizes ``--src`` (the originals, with caption sidecars) into
         that subset's ``image_dir`` — the source dir is never modified;
      2. caches VAE latents from ``image_dir`` into the subset's ``cache_dir``;
      3. caches text embeddings (captions read from ``--src``) into ``cache_dir``.

    A config can't encode where the *un-resized* originals live (its
    ``image_dir`` is the post-resize dir training reads), so the source is the
    one explicit flag: ``--src <dir>``. The ComfyUI trainer node uses this to
    cache a single-image temp dir before its chained training job runs.

    The VAE / text-encoder / DiT used for caching default to the config-resolved
    ``models/`` paths (base → preset → method merge), but can be overridden with
    ``--vae`` / ``--qwen3`` / ``--dit`` so a caller can point the cache at models
    living elsewhere — e.g. the ComfyUI trainer node passes the paths ComfyUI's
    own ``folder_paths`` registers, so it never assumes a copy under
    ``anima_lora/models/``.

    Usage: ``preprocess-config --dataset_config <path> --src <dir>
    [--vae <path>] [--qwen3 <path>] [--dit <path>] [extra…]``
    (any remaining args are forwarded to the resize step).
    """
    import toml

    args = list(extra)
    cfg_path: str | None = None
    src_dir: str | None = None
    vae_path = _path("vae", "models/vae/qwen_image_vae.safetensors")
    qwen3_path = _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors")
    dit_path = _path(
        "pretrained_model_name_or_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--dataset_config" and i + 1 < len(args):
            cfg_path = args[i + 1]
            i += 2
        elif args[i] == "--src" and i + 1 < len(args):
            src_dir = args[i + 1]
            i += 2
        elif args[i] == "--vae" and i + 1 < len(args):
            vae_path = args[i + 1]
            i += 2
        elif args[i] == "--qwen3" and i + 1 < len(args):
            qwen3_path = args[i + 1]
            i += 2
        elif args[i] == "--dit" and i + 1 < len(args):
            dit_path = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    if not cfg_path or not src_dir:
        raise SystemExit(
            "preprocess-config requires --dataset_config <path> and --src <dir>"
        )

    # A real-time scanner (e.g. Windows Defender) often holds a brief exclusive
    # lock on a *just-created* file, surfaced as PermissionError [Errno 13] on
    # Windows. The ComfyUI trainer node writes this config milliseconds before
    # the daemon's preprocess job opens it, so retry through that transient lock
    # rather than failing the whole chain. A genuinely unreadable file still
    # raises after the budget is spent.
    import time

    last_err: OSError | None = None
    for attempt in range(10):
        try:
            cfg = toml.load(cfg_path)
            break
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    else:
        raise SystemExit(
            f"could not read {cfg_path} after retrying (last error: {last_err}). "
            "If this persists, exclude the dataset/temp dir from your antivirus."
        )
    subsets = [
        sub
        for ds in (cfg.get("datasets") or [])
        for sub in (ds.get("subsets") or [])
        if sub.get("image_dir")
    ]
    if not subsets:
        raise SystemExit(f"no [[datasets.subsets]] with image_dir in {cfg_path}")

    for sub in subsets:
        image_dir = sub["image_dir"]
        cache_dir = sub.get("cache_dir") or image_dir
        # 1) bucket-resize originals → image_dir. cache_latents.py keys caches
        #    by the on-disk (native) size, so the resized size must already be
        #    the constant-token bucket the trainer will select. Captions stay
        #    in --src (TE caching reads them there).
        run(
            [
                PY,
                "scripts/preprocess/resize_images.py",
                "--src",
                src_dir,
                "--dst",
                image_dir,
                "--no_copy_captions",
                "--min_pixels",
                "0",
                "--bucket_reso_steps",
                "64",
                "--recursive",
                *rest,
            ]
        )
        # 2) VAE latents → cache_dir
        run(
            [
                PY,
                "scripts/preprocess/cache_latents.py",
                "--dir",
                image_dir,
                "--cache_dir",
                cache_dir,
                "--vae",
                vae_path,
                "--batch_size",
                "4",
                "--chunk_size",
                "64",
                "--recursive",
            ]
        )
        # 3) text embeddings (captions read from --src) → cache_dir
        run(
            [
                PY,
                "scripts/preprocess/cache_text_embeddings.py",
                "--dir",
                src_dir,
                "--cache_dir",
                cache_dir,
                "--qwen3",
                qwen3_path,
                "--dit",
                dit_path,
                "--recursive",
            ]
        )
