from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.tasks.masking import _detect_subset_dirs


class TestDetectSubsetDirs:
    def test_detects_subdirs_with_resized(self, tmp_path: Path) -> None:
        (tmp_path / "4_a" / ".resized").mkdir(parents=True)
        (tmp_path / "5_b" / ".resized").mkdir(parents=True)

        result = _detect_subset_dirs(tmp_path)

        names = [name for name, _ in result]
        assert names == ["4_a", "5_b"]
        assert result[0][1] == tmp_path / "4_a" / ".resized"
        assert result[1][1] == tmp_path / "5_b" / ".resized"

    def test_detects_root_resized(self, tmp_path: Path) -> None:
        (tmp_path / ".resized").mkdir()

        result = _detect_subset_dirs(tmp_path)

        assert len(result) == 1
        assert result[0] == ("", tmp_path / ".resized")

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden" / ".resized").mkdir(parents=True)
        (tmp_path / "visible" / ".resized").mkdir(parents=True)

        result = _detect_subset_dirs(tmp_path)

        names = [name for name, _ in result]
        assert names == ["visible"]

    def test_empty_returns_empty_list(self, tmp_path: Path) -> None:
        result = _detect_subset_dirs(tmp_path)
        assert result == []

    def test_mixed_root_and_subdirs(self, tmp_path: Path) -> None:
        (tmp_path / ".resized").mkdir()
        (tmp_path / "3_a" / ".resized").mkdir(parents=True)

        result = _detect_subset_dirs(tmp_path)

        names = [name for name, _ in result]
        assert names == ["", "3_a"]

    def test_skips_files(self, tmp_path: Path) -> None:
        (tmp_path / "a_file.txt").write_text("not a dir")
        (tmp_path / "real" / ".resized").mkdir(parents=True)

        result = _detect_subset_dirs(tmp_path)

        names = [name for name, _ in result]
        assert names == ["real"]


class TestResolveMaskDirTreeMode:
    def test_per_subset_masks_highest_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subset_masks = tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".masks"
        subset_masks.mkdir(parents=True)
        (tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".resized").mkdir(
            parents=True
        )
        global_masks = tmp_path / "post_image_dataset" / "masks"
        global_masks.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)

        from library.datasets.subsets import _resolve_default_mask_dir

        image_dir = str(tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".resized")
        result = _resolve_default_mask_dir(image_dir=image_dir)

        assert result == os.path.join(
            str(tmp_path / "post_image_dataset" / "mychar" / "4_a"), ".masks"
        )

    def test_fallback_to_global_when_no_subset_masks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".resized").mkdir(
            parents=True
        )
        global_masks = tmp_path / "post_image_dataset" / "masks"
        global_masks.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)

        from library.datasets.subsets import _resolve_default_mask_dir

        image_dir = str(tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".resized")
        result = _resolve_default_mask_dir(image_dir=image_dir)

        assert result == "post_image_dataset/masks"

    def test_no_image_dir_backward_compatible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_masks = tmp_path / "post_image_dataset" / "masks"
        global_masks.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)

        from library.datasets.subsets import _resolve_default_mask_dir

        result = _resolve_default_mask_dir(image_dir=None)

        assert result == "post_image_dataset/masks"

    def test_legacy_masks_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legacy = tmp_path / "masks" / "merged"
        legacy.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)

        from library.datasets.subsets import _resolve_default_mask_dir

        result = _resolve_default_mask_dir(image_dir=None)

        assert result == "masks/merged"

    def test_returns_none_when_nothing_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        from library.datasets.subsets import _resolve_default_mask_dir

        result = _resolve_default_mask_dir(image_dir=None)

        assert result is None


def _resolve_mask_path(
    image_path: Path, current_dir: Path | None, root: Path
) -> Path | None:
    if current_dir is None:
        return None
    try:
        rel = image_path.relative_to(current_dir)
    except ValueError:
        return None
    rel_parent = rel.parent
    name = f"{image_path.stem}_mask.png"

    parent_dir = image_path.parent
    if parent_dir.name == ".resized":
        subset_masks = parent_dir.parent / ".masks" / name
    else:
        subset_masks = parent_dir / ".masks" / name
    if subset_masks.is_file():
        return subset_masks

    for candidate_root in (
        root / "post_image_dataset" / "masks",
        root / "masks" / "merged",
    ):
        candidate = candidate_root / rel_parent / name
        if candidate.is_file():
            return candidate

    return None


class TestResolveMaskPathGui:
    def test_finds_subset_masks(self, tmp_path: Path) -> None:
        resized_dir = tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".resized"
        resized_dir.mkdir(parents=True)
        masks_dir = tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".masks"
        masks_dir.mkdir(parents=True)

        image_path = resized_dir / "img001.png"
        image_path.write_bytes(b"\x89PNG")
        mask_path = masks_dir / "img001_mask.png"
        mask_path.write_bytes(b"\x89PNG")

        result = _resolve_mask_path(
            image_path,
            tmp_path / "post_image_dataset" / "mychar" / "4_a",
            tmp_path,
        )

        assert result == mask_path

    def test_fallback_to_global_masks(self, tmp_path: Path) -> None:
        resized_dir = tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".resized"
        resized_dir.mkdir(parents=True)
        global_masks = tmp_path / "post_image_dataset" / "masks" / ".resized"
        global_masks.mkdir(parents=True)

        image_path = resized_dir / "img001.png"
        image_path.write_bytes(b"\x89PNG")
        mask_path = global_masks / "img001_mask.png"
        mask_path.write_bytes(b"\x89PNG")

        result = _resolve_mask_path(
            image_path,
            tmp_path / "post_image_dataset" / "mychar" / "4_a",
            tmp_path,
        )

        assert result == mask_path

    def test_returns_none_when_no_mask(self, tmp_path: Path) -> None:
        resized_dir = tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".resized"
        resized_dir.mkdir(parents=True)

        image_path = resized_dir / "img001.png"
        image_path.write_bytes(b"\x89PNG")

        result = _resolve_mask_path(
            image_path,
            tmp_path / "post_image_dataset" / "mychar" / "4_a",
            tmp_path,
        )

        assert result is None

    def test_returns_none_when_no_current_dir(self, tmp_path: Path) -> None:
        image_path = tmp_path / "some" / "img001.png"

        result = _resolve_mask_path(image_path, None, tmp_path)

        assert result is None

    def test_non_resized_dir_uses_parent_masks(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "post_image_dataset" / "mychar" / "4_a"
        img_dir.mkdir(parents=True)
        masks_dir = tmp_path / "post_image_dataset" / "mychar" / "4_a" / ".masks"
        masks_dir.mkdir(parents=True)

        image_path = img_dir / "img001.png"
        image_path.write_bytes(b"\x89PNG")
        mask_path = masks_dir / "img001_mask.png"
        mask_path.write_bytes(b"\x89PNG")

        result = _resolve_mask_path(
            image_path,
            tmp_path / "post_image_dataset" / "mychar",
            tmp_path,
        )

        assert result == mask_path
