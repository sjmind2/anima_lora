import math
import random
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np

# Bucket resolutions as (W, H), grouped into six token-count families: 4032
# (= 63*64), 4200 (= 60*70), 4096 (= 64*64), 6144 (= 96*64), 9216 (= 96*96),
# and 1024 (= 32*32). The first two are highly composite, so each factors into
# many near-square→elongated patch grids — and crucially every bucket *exactly*
# fills its token count, so there is zero intra-bucket padding by construction.
#
# This table is designed for native shapes (the only mode): it collapses to
# up to SIX distinct token counts → six compiled block graphs (via
# compile_blocks' flatten), with no padding and therefore no flash pad leak.
# The rope per-axis cap is 256 patches (max_img/patch_spatial); the largest dim
# here is 2016px → 126.
#
# The first two families densely cover aspect space; a single token count's
# divisors near √N are sparse (4032 alone jumps aspect 1.29→1.75), so
# interleaving 4032 and 4200 densely covers aspect space at the cost of one
# extra graph. The additional families (4096, 6144, 9216, 1024) provide
# exact-match buckets for common resolutions. Landscape mirrors (swap W, H)
# are included explicitly. Token count = (W//16)*(H//16).
#
# NOTE: DCW_ASPECT_BUCKETS below now draws its top-5 from this table (every
# entry is a real training bucket), so `make dcw` recalibration produces rows
# for every aspect_id. Do not reorder the DCW table (shipped fusion-head
# checkpoints key off it).
CONSTANT_TOKEN_BUCKETS = [
    # ---- 4032-token family (63*64) ----
    (1008, 1024),  # 63 x 64, ar 0.98 (nearest to square)
    (1024, 1008),  #          ar 1.02
    (896, 1152),  # 56 x 72, ar 0.78
    (1152, 896),  #          ar 1.29
    (768, 1344),  # 48 x 84, ar 0.57
    (1344, 768),  #          ar 1.75
    (672, 1536),  # 42 x 96, ar 0.44
    (1536, 672),  #          ar 2.29
    (576, 1792),  # 36 x 112, ar 0.32
    (1792, 576),  #           ar 3.11
    (512, 2016),  # 32 x 126, ar 0.25
    (2016, 512),  #           ar 3.94
    # ---- 4200-token family (60*70) ----
    (960, 1120),  # 60 x 70, ar 0.86
    (1120, 960),  #          ar 1.17
    (896, 1200),  # 56 x 75, ar 0.75
    (1200, 896),  #          ar 1.34
    (800, 1344),  # 50 x 84, ar 0.60
    (1344, 800),  #          ar 1.68
    (672, 1600),  # 42 x 100, ar 0.42
    (1600, 672),  #           ar 2.38
    (640, 1680),  # 40 x 105, ar 0.38
    (1680, 640),  #           ar 2.62
    (560, 1920),  # 35 x 120, ar 0.29
    (1920, 560),  #           ar 3.43
    # ---- 4096-token family (64*64) ----
    (1024, 1024),  # 64 x 64, ar 1.00 (exact square)
    # ---- 1024-token family (32*32) ----
    (512, 512),  # 32 x 32, ar 1.00 (small square)
]

BUCKET_FAMILIES = {
    "XL": {
        "tc": 5040,
        "members": [
            (640, 2016),
            (672, 1920),
            (720, 1792),
            (768, 1680),
            (896, 1440),
            (960, 1344),
            (1008, 1280),
            (1120, 1152),
        ],
    },
    "L": {
        "tc": 4032,
        "members": [
            (512, 2016),
            (576, 1792),
            (672, 1536),
            (768, 1344),
            (896, 1152),
            (1008, 1024),
        ],
    },
    "M": {
        "tc": 3600,
        "members": [
            (480, 1920),
            (576, 1600),
            (640, 1440),
            (720, 1280),
            (768, 1200),
            (800, 1152),
            (960, 960),
        ],
    },
    "S": {
        "tc": 2160,
        "members": [
            (384, 1440),
            (432, 1280),
            (480, 1152),
            (576, 960),
            (640, 864),
            (720, 768),
        ],
    },
    "XS": {
        "tc": 1680,
        "members": [
            (336, 1280),
            (384, 1120),
            (448, 960),
            (480, 896),
            (560, 768),
            (640, 672),
        ],
    },
    "S1": {"tc": 1024, "members": [(256, 1024), (512, 512)]},
    "S2": {"tc": 4096, "members": [(512, 2048), (1024, 1024)]},
}


def get_bucket_list(enabled_families=None):
    if enabled_families is None:
        enabled_families = list(BUCKET_FAMILIES.keys())
    buckets = []
    seen = set()
    for name in enabled_families:
        if name not in BUCKET_FAMILIES:
            continue
        for W, H in BUCKET_FAMILIES[name]["members"]:
            if (W, H) not in seen:
                buckets.append((W, H))
                seen.add((W, H))
            if W != H and (H, W) not in seen:
                buckets.append((H, W))
                seen.add((H, W))
    return buckets


# DCW v4 calibration aspect-bucket set.
#
# Top 5 (H, W) resolutions by frequency in post_image_dataset/lora/ (recounted
# 2026-05-23; every entry is a CONSTANT_TOKEN_BUCKETS training bucket). List
# order *is* the canonical aspect_id index — DCW v4's per-aspect statistics
# (fusion_head.safetensors per-bucket μ_g, σ²_prior, λ_scalar) key off this
# order, so a reorder invalidates every shipped fusion-head checkpoint.
#
# Read by both the calibration data-gen path (scripts/tasks/dcw.py drives
# `make dcw` over these buckets) and the fusion-head trainer
# (scripts/dcw/fusion_data.py uses the dict for the (H, W) → aspect_id
# lookup that decides which run rows feed the trainer). Inference itself
# is bucket-agnostic post-cleanup — see project_dcw_bucket_prior_cosmetic.
DCW_ASPECT_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1200, 896),  # 0 — 896x1200 portrait (most common, 4200-tok)
    (1344, 800),  # 1 — 800x1344 tall portrait (4200-tok)
    (896, 1200),  # 2 — 1200x896 landscape (4200-tok)
    (1344, 768),  # 3 — 768x1344 tall portrait (4032-tok)
    (1152, 896),  # 4 — 896x1152 portrait (4032-tok)
)
DCW_ASPECT_NAMES: Tuple[str, ...] = tuple(f"{h}x{w}" for h, w in DCW_ASPECT_BUCKETS)
DCW_ASPECT_TABLE: dict = {hw: i for i, hw in enumerate(DCW_ASPECT_BUCKETS)}
N_DCW_ASPECTS: int = len(DCW_ASPECT_BUCKETS)


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    """Generate bucket resolutions for multi-aspect-ratio training.
    Moved from model_util.py to avoid dependency."""
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


class BucketManager:
    def __init__(
        self, max_reso=None, min_size=None, max_size=None, reso_steps=None
    ) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], (
                    "the max_size should be larger than the width of max_reso"
                )
                assert max_size >= max_reso[1], (
                    "the max_size should be larger than the height of max_reso"
                )
            if min_size is not None:
                assert max_size >= min_size, (
                    "the max_size should be larger than the min_size"
                )

        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self, constant_token_buckets: bool = False, enabled_families=None):
        if enabled_families is not None:
            resos = get_bucket_list(enabled_families)
            family_groups = {}
            for name in enabled_families:
                if name not in BUCKET_FAMILIES:
                    continue
                members = []
                for W, H in BUCKET_FAMILIES[name]["members"]:
                    members.append((W, H))
                    if W != H:
                        members.append((H, W))
                family_groups[name] = members
            self.set_predefined_resos(resos, family_groups=family_groups)
        elif constant_token_buckets:
            resos = list(CONSTANT_TOKEN_BUCKETS)
            self.set_predefined_resos(resos)
        else:
            resos = make_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size, self.reso_steps
            )
            self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos, family_groups=None):
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])
        self.family_groups = family_groups
        if family_groups:
            self.family_tc = {
                name: BUCKET_FAMILIES[name]["tc"]
                for name in family_groups
                if name in BUCKET_FAMILIES
            }
        else:
            self.family_tc = None

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        image_area = image_width * image_height

        if self.family_groups:
            best_family = min(
                self.family_tc.items(),
                key=lambda kv: abs(kv[1] * 256 - image_area),
            )[0]
            family_resos = self.family_groups[best_family]
            family_ars = np.array([w / h for w, h in family_resos])
            ar_errors = np.abs(family_ars - aspect_ratio)
            min_err = ar_errors.min()
            tied = np.where(ar_errors == min_err)[0]
            if len(tied) > 1:
                areas = np.array([w * h for w, h in family_resos])[tied]
                idx = tied[np.abs(areas - image_area).argmin()]
            else:
                idx = tied[0]
            reso = family_resos[idx]
        else:
            reso = (image_width, image_height)
            if reso not in self.predefined_resos_set:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                abs_ar_errors = np.abs(ar_errors)
                min_ar_error = abs_ar_errors.min()
                tied = np.where(abs_ar_errors == min_ar_error)[0]
                if len(tied) > 1:
                    areas = np.array([w * h for w, h in self.predefined_resos])[tied]
                    predefined_bucket_id = tied[np.abs(areas - image_area).argmin()]
                else:
                    predefined_bucket_id = tied[0]
                reso = self.predefined_resos[predefined_bucket_id]

        ar_reso = reso[0] / reso[1]
        if aspect_ratio > ar_reso:
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width

        resized_size = (
            int(image_width * scale + 0.5),
            int(image_height * scale + 0.5),
        )

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


IMAGE_EXTS_SCAN = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"})


def scan_dataset_bucket_distribution(
    source_dir: str, enabled_families: list[str]
) -> dict:
    src = Path(source_dir)
    if not src.is_dir():
        return {"error": "Directory not found"}

    all_tc = {name: info["tc"] for name, info in BUCKET_FAMILIES.items()}
    enabled_tc = {name: all_tc[name] for name in enabled_families if name in all_tc}

    original_counts: dict[str, int] = {name: 0 for name in all_tc}
    resized_counts: dict[str, int] = {name: 0 for name in all_tc}

    total = 0
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS_SCAN:
            continue
        try:
            from PIL import Image

            with Image.open(p) as img:
                iw, ih = img.size
        except Exception:
            continue
        total += 1
        img_area = iw * ih

        best_all = min(all_tc.items(), key=lambda kv: abs(kv[1] * 256 - img_area))[0]
        original_counts[best_all] += 1

        if enabled_tc:
            best_enabled = min(
                enabled_tc.items(), key=lambda kv: abs(kv[1] * 256 - img_area)
            )[0]
            resized_counts[best_enabled] += 1
        else:
            resized_counts[best_all] += 1

    return {
        "total_images": total,
        "families": {
            name: {
                "original": original_counts[name],
                "resized": resized_counts[name],
            }
            for name in BUCKET_FAMILIES
        },
    }
