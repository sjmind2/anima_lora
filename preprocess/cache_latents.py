#!/usr/bin/env python3
"""Cache VAE latents for all images in a dataset directory.

Encodes images through the Qwen Image VAE and saves latent caches (.npz)
alongside the images.  Skips already-cached entries (idempotent).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.io.cache import LATENT_CACHE_SUFFIX
from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS


def get_latents_npz_path(
    image_path: Path,
    image_size: tuple[int, int],
    cache_dir: Path | None = None,
) -> Path:
    """Match the naming convention used by AnimaLatentsCachingStrategy.

    When ``cache_dir`` is provided, the cache lives under that directory with
    a stem-mirrored filename instead of as a sidecar.
    """
    name = (
        f"{image_path.stem}_{image_size[0]:04d}x{image_size[1]:04d}"
        f"{LATENT_CACHE_SUFFIX}"
    )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / name
    return image_path.with_name(name)


def _cache_for_images(
    image_files: list[Path],
    cache_dir: Path | None,
    vae,
    device: torch.device,
    dtype,
    batch_size: int,
    label: str = "Caching latents",
) -> tuple[int, int]:
    reso_groups: dict[tuple[int, int], list[Path]] = {}
    for p in image_files:
        img = Image.open(p)
        size = img.size
        img.close()
        reso_groups.setdefault(size, []).append(p)

    total = len(image_files)
    cached = 0
    skipped = 0

    pbar = tqdm(total=total, desc=label)
    for (w, h), paths in reso_groups.items():
        for batch_start in range(0, len(paths), batch_size):
            batch_paths = paths[batch_start : batch_start + batch_size]
            tensors = []

            for p in batch_paths:
                npz_path = get_latents_npz_path(p, (w, h), cache_dir=cache_dir)
                if npz_path.exists():
                    latents_size = (h // 8, w // 8)
                    key = f"latents_{latents_size[0]}x{latents_size[1]}"
                    try:
                        npz = np.load(npz_path)
                        if key in npz:
                            skipped += 1
                            pbar.update(1)
                            pbar.set_postfix_str(f"skip {p.name}")
                            continue
                    except Exception:
                        pass

                img = Image.open(p).convert("RGB")
                img_np = np.array(img)
                img_tensor = IMAGE_TRANSFORMS(img_np)
                tensors.append((p, img_tensor, (w, h)))

            if not tensors:
                continue

            img_batch = torch.stack([t[1] for t in tensors], dim=0)
            img_batch = img_batch.to(device=device, dtype=dtype)

            with torch.no_grad():
                latents = vae.encode_pixels_to_latents(img_batch).cpu()

            for i, (p, _, size) in enumerate(tensors):
                lat = latents[i]
                latents_size = lat.shape[-2:]
                key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"

                npz_path = get_latents_npz_path(p, size, cache_dir=cache_dir)
                kwargs = {}
                if npz_path.exists():
                    npz = np.load(npz_path)
                    for key in npz.files:
                        kwargs[key] = npz[key]

                kwargs[f"latents{key_reso_suffix}"] = lat.float().numpy()
                kwargs[f"original_size{key_reso_suffix}"] = np.array(list(size))
                kwargs[f"crop_ltrb{key_reso_suffix}"] = np.array(
                    [0, 0, size[0], size[1]]
                )

                np.savez(npz_path, **kwargs)

                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{p.name} → {size[0]}x{size[1]}")

    pbar.close()
    return cached, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, required=True, help="Dataset directory")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to write latent caches into (created if needed). "
            "Defaults to writing alongside each source image."
        ),
    )
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE weights")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="VAE encoding batch size (default: 4)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="VAE spatial chunk size (default: 64)",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        default=True,
        help="Disable VAE internal cache (default: True)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk subfolders under --dir. Caches are still written flat "
            "(stem-based filenames); image stems must therefore be unique "
            "across the entire source tree."
        ),
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help=(
            "Tree mode: scan --dir for subdirectories containing .resized/ "
            "and process each independently. Caches go to .lora/ next to each "
            ".resized/ directory."
        ),
    )
    args = parser.parse_args()

    from library.models import qwen_vae as qwen_image_autoencoder_kl

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading VAE from {args.vae} ...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.chunk_size,
        disable_cache=args.disable_cache,
    )
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    if args.tree:
        resized_dirs: list[tuple[Path, Path]] = []

        root_resized = data_dir / ".resized"
        if root_resized.is_dir():
            resized_dirs.append((root_resized, data_dir / ".lora"))

        for child in sorted(data_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            resized = child / ".resized"
            if resized.is_dir():
                resized_dirs.append((resized, child / ".lora"))

        total_cached = 0
        total_skipped = 0
        for resized_dir, subset_cache_dir in resized_dirs:
            print(f"\n--- {resized_dir} → {subset_cache_dir} ---")
            image_files = sorted(
                p
                for p in resized_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_files:
                print("  (no images found, skipping)")
                continue

            stems: dict[str, Path] = {}
            collisions: list[tuple[str, Path, Path]] = []
            for p in image_files:
                if p.stem in stems:
                    collisions.append((p.stem, stems[p.stem], p))
                else:
                    stems[p.stem] = p
            if collisions:
                print("Duplicate image stems found (caches are stem-keyed):")
                for stem, a, b in collisions:
                    print(f"  '{stem}': {a} <-> {b}")
                sys.exit(1)

            cached, skipped = _cache_for_images(
                image_files,
                subset_cache_dir,
                vae,
                device,
                dtype,
                args.batch_size,
                label=resized_dir.parent.name or str(data_dir),
            )
            print(f"  {cached} cached, {skipped} skipped (already existed)")
            total_cached += cached
            total_skipped += skipped

        print(
            f"\nTree complete: {total_cached} cached, {total_skipped} skipped"
        )
    else:
        if args.recursive:
            image_files = sorted(
                p
                for p in data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            stems: dict[str, Path] = {}
            collisions: list[tuple[str, Path, Path]] = []
            for p in image_files:
                if p.stem in stems:
                    collisions.append((p.stem, stems[p.stem], p))
                else:
                    stems[p.stem] = p
            if collisions:
                print("Duplicate image stems found under --dir (caches are stem-keyed):")
                for stem, a, b in collisions:
                    print(f"  '{stem}': {a} <-> {b}")
                sys.exit(1)
        else:
            image_files = sorted(
                p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )

        cached, skipped = _cache_for_images(
            image_files, cache_dir, vae, device, dtype, args.batch_size
        )
        print(
            f"\nLatent caching complete: {cached} cached, {skipped} skipped (already existed)"
        )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
