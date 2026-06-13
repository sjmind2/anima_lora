"""Anima LoRA — PySide6 GUI package.

The package root is a thin facade: path constants live in :mod:`gui._paths`,
and the bulk of what used to live here was split into focused submodules —

* :mod:`gui.config_io`   — variant/preset discovery, load/save, merge, lint (Qt-free)
* :mod:`gui.validation`  — validation-split encoding (Qt-free)
* :mod:`gui.dialogs`     — resume/cache confirmation popups + on-disk probes
* :mod:`gui.discovery`   — image/adapter/dataset directory walks (Qt-free)
* :mod:`gui.widgets`     — LazyTabMixin, the config-form field factory, ScaledImageLabel

Everything is re-exported here so the historical ``from gui import <name>``
call sites in the tabs keep working unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import toml

logger = logging.getLogger(__name__)
from PySide6.QtWidgets import QMessageBox

from gui._paths import (
    CONFIGS_DIR,
    CUSTOM_DIR,
    CUSTOM_VARIANTS_DIR,
    GUI_METHODS_DIR,
    IMAGE_EXTS,
    METHODS_DIR,
    PRESETS_FILE,
    ROOT,
)
from gui.config_io import (
    _BASIC,
    _GROUPS,
    _K2G,
    _SKIP,
    _VIRTUAL_KEYS,
    _builtin_variants_by_family,
    _dataset_lint_sources,
    _load,
    _load_all_presets,
    _load_base,
    _read_variant_metadata,
    _save,
    custom_preset_path,
    custom_variant_path,
    is_basic_field,
    is_custom_preset,
    is_custom_variant,
    lint_variant_configs,
    list_gui_variants,
    list_methods,
    list_presets,
    merged_gui_variant_preset,
    merged_method_preset,
    remove_unknown_dataset_keys,
    variant_metadata,
    variant_path,
)
from gui.dialogs import (
    confirm_existing_caches,
    confirm_resumable_checkpoint,
    confirm_train_using_cache,
    count_preprocess_caches,
    find_resumable_checkpoint,
)
from gui.discovery import (
    _adapter_dirs,
    _image_dirs,
    _imgs,
    _safetensors_in,
)
from gui.validation import (
    _base_folder_repeats,
    apply_folder_repeats_choice,
    apply_validation_choice,
)
from gui.widgets import (
    LazyTabMixin,
    ScaledImageLabel,
    _SamplePromptsWidget,
    _no_wheel,
    _read,
    _TargetResWidget,
    _widget,
)

# Cache-file suffixes written by the preprocess scripts. Kept in sync with
# scripts/preprocess/cache_latents.py, cache_text_embeddings.py, cache_pe_encoder.py.
_LATENT_SUFFIX = "_anima.npz"
_TE_SUFFIX = "_anima_te.safetensors"
_PE_SUFFIX = "_anima_pe.safetensors"


# ── Fork-specific: source dir scanning + bucket manifest ──────────────────


def scan_source_dir(source_dir: str, *, is_reg: bool = False) -> list[dict]:
    logger.info("scan_source_dir: scanning source_dir=%r", source_dir)
    p = Path(source_dir)
    if not p.is_dir():
        logger.warning(
            "scan_source_dir: %r is not a directory or does not exist, returning empty list",
            source_dir,
        )
        return []
    basename = p.name
    logger.debug("scan_source_dir: resolved basename=%r", basename)
    subsets = []
    root_images = [
        f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    logger.debug(
        "scan_source_dir: found %d image file(s) in root directory", len(root_images)
    )
    if root_images:
        sub = {
            "name": "(root)",
            "source_dir": str(p),
            "image_dir": f"post_image_dataset/{basename}/.resized",
            "cache_dir": f"post_image_dataset/{basename}/.lora",
            "num_repeats": 1,
            "recursive": True,
            "is_reg": is_reg,
        }
        subsets.append(sub)
        logger.info(
            "scan_source_dir: added (root) subset  image_dir=%r  cache_dir=%r  num_repeats=1  is_reg=%s",
            sub["image_dir"],
            sub["cache_dir"],
            is_reg,
        )
    skipped = []
    for child in sorted(p.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            skipped.append(child.name)
            continue
        m = re.match(r"^(\d+)_.+", child.name)
        num_repeats = int(m.group(1)) if m else 1
        sub = {
            "name": child.name,
            "source_dir": str(child),
            "image_dir": f"post_image_dataset/{basename}/{child.name}/.resized",
            "cache_dir": f"post_image_dataset/{basename}/{child.name}/.lora",
            "num_repeats": num_repeats,
            "recursive": True,
            "is_reg": is_reg,
        }
        subsets.append(sub)
        if m:
            logger.info(
                "scan_source_dir: subset=%r  matched pattern [N]_rest → num_repeats=%d  image_dir=%r  cache_dir=%r  is_reg=%s",
                child.name,
                num_repeats,
                sub["image_dir"],
                sub["cache_dir"],
                is_reg,
            )
        else:
            logger.info(
                "scan_source_dir: subset=%r  no repeat prefix → num_repeats=%d (default)  image_dir=%r  cache_dir=%r  is_reg=%s",
                child.name,
                num_repeats,
                sub["image_dir"],
                sub["cache_dir"],
                is_reg,
            )
    if skipped:
        logger.debug("scan_source_dir: skipped hidden directories: %s", skipped)
    logger.info(
        "scan_source_dir: total %d subset(s) generated for source_dir=%r",
        len(subsets),
        source_dir,
    )
    return subsets


def find_stale_latent_caches(cache_dir: Path, enabled_families=None) -> dict[str, int]:
    """Return a ``{"WxH": count}`` map of VAE latent caches whose pixel
    resolution is NOT in the live bucket table."""
    if not cache_dir.is_dir():
        return {}
    from library.datasets.buckets import get_bucket_list

    families = enabled_families if enabled_families is not None else None
    valid = {f"{w}x{h}" for (w, h) in get_bucket_list(families)}
    stale: dict[str, int] = {}
    for p in cache_dir.rglob("*"):
        if not p.is_file() or not p.name.endswith(_LATENT_SUFFIX):
            continue
        tail = p.name.removesuffix(_LATENT_SUFFIX).rsplit("_", 1)
        if len(tail) < 2:
            continue
        m = re.fullmatch(r"(\d+)x(\d+)", tail[1])
        if not m:
            continue
        key = f"{int(m.group(1))}x{int(m.group(2))}"
        if key not in valid:
            stale[key] = stale.get(key, 0) + 1
    return stale


def confirm_stale_caches(
    parent, cache_dir: Path, enabled_families=None
) -> bool:
    """Warn if any VAE latent cache sits at a resolution outside the current
    bucket table."""
    from gui.i18n import t

    stale = find_stale_latent_caches(cache_dir, enabled_families=enabled_families)
    if not stale:
        return True
    total = sum(stale.values())
    shown = sorted(stale.items(), key=lambda kv: -kv[1])[:6]
    examples = "\n".join(f"  • {reso}  ({n}×)" for reso, n in shown)
    if len(stale) > len(shown):
        examples += "\n  • …"
    body = t("stale_cache_body", n=total, cache_dir=str(cache_dir), examples=examples)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(t("stale_cache_title"))
    box.setText(body)
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Cancel)
    return box.exec() == QMessageBox.Yes


def write_bucket_manifest(cache_dir: Path, enabled_families: list[str]) -> None:
    from library.datasets.buckets import BUCKET_FAMILIES, get_bucket_list

    manifest = {
        "version": 1,
        "enabled_families": enabled_families,
        "buckets": get_bucket_list(enabled_families),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / ".bucket_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def check_bucket_manifest(cache_dir: Path, enabled_families: list[str]) -> bool:
    from library.datasets.buckets import get_bucket_list

    manifest_path = cache_dir / ".bucket_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    saved = set(tuple(b) for b in data.get("buckets", []))
    current = set(tuple(b) for b in get_bucket_list(enabled_families))
    return saved == current


def confirm_bucket_mismatch(
    parent, cache_dir: Path, enabled_families: list[str]
) -> str | None:
    if check_bucket_manifest(cache_dir, enabled_families):
        return "ok"
    manifest_path = cache_dir / ".bucket_manifest.json"
    if not manifest_path.exists():
        return "missing"
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Bucket Configuration Mismatch")
    msg.setText("The cached bucket configuration does not match the current selection.")
    msg.setInformativeText("Choose an action:")
    recalc = msg.addButton("Recalculate", QMessageBox.ButtonRole.AcceptRole)
    skip = msg.addButton("Skip (use existing)", QMessageBox.ButtonRole.RejectRole)
    cancel = msg.addButton("Cancel", QMessageBox.ButtonRole.DestructiveRole)
    msg.exec()
    clicked = msg.clickedButton()
    if clicked == recalc:
        return "recalc"
    elif clicked == skip:
        return "skip"
    else:
        return "cancel"


def load_bucket_families() -> list[str]:
    from library.datasets.buckets import BUCKET_FAMILIES

    settings_file = Path(__file__).resolve().parent / "gui_settings.json"
    default = ["M", "L"]
    if not settings_file.exists():
        return default
    try:
        data = json.loads(settings_file.read_text(encoding="utf-8"))
    except Exception:
        return default
    families = data.get("bucket_families", default)
    return [f for f in families if f in BUCKET_FAMILIES]


def scan_images_for_bucket_stats(
    source_dir: str, enabled_families: list[str]
) -> dict[str, int]:
    from library.datasets.buckets import scan_dataset_bucket_distribution

    result = scan_dataset_bucket_distribution(source_dir, enabled_families)
    if "error" in result:
        return {}
    return {name: info["resized"] for name, info in result["families"].items()}


__all__ = [
    "ROOT",
    "CONFIGS_DIR",
    "IMAGE_EXTS",
    "METHODS_DIR",
    "GUI_METHODS_DIR",
    "PRESETS_FILE",
    "CUSTOM_DIR",
    "CUSTOM_VARIANTS_DIR",
    "LazyTabMixin",
    "ScaledImageLabel",
    "_SamplePromptsWidget",
    "_TargetResWidget",
    "_no_wheel",
    "_read",
    "_widget",
    "_load",
    "_load_base",
    "_save",
    "_load_all_presets",
    "_builtin_variants_by_family",
    "_read_variant_metadata",
    "_dataset_lint_sources",
    "_GROUPS",
    "_K2G",
    "_SKIP",
    "_BASIC",
    "_VIRTUAL_KEYS",
    "is_basic_field",
    "list_methods",
    "list_gui_variants",
    "list_presets",
    "is_custom_variant",
    "is_custom_preset",
    "custom_variant_path",
    "custom_preset_path",
    "variant_path",
    "variant_metadata",
    "lint_variant_configs",
    "remove_unknown_dataset_keys",
    "merged_method_preset",
    "merged_gui_variant_preset",
    "apply_validation_choice",
    "apply_folder_repeats_choice",
    "_base_folder_repeats",
    "confirm_resumable_checkpoint",
    "confirm_existing_caches",
    "confirm_train_using_cache",
    "count_preprocess_caches",
    "find_resumable_checkpoint",
    "confirm_stale_caches",
    "find_stale_latent_caches",
    "write_bucket_manifest",
    "check_bucket_manifest",
    "confirm_bucket_mismatch",
    "load_bucket_families",
    "scan_images_for_bucket_stats",
    "scan_source_dir",
    "_imgs",
    "_safetensors_in",
    "_adapter_dirs",
    "_image_dirs",
    "main",
]


def main():
    from gui.app import main as _main

    _main()
