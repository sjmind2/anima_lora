from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest


CACHE_SUFFIX = "_anima.npz"


def _build_search_dirs(cache_dir: str | None, image_dir: str | None) -> list[str]:
    npz_search_dirs: list[str] = []
    if cache_dir and os.path.isdir(cache_dir):
        npz_search_dirs.append(cache_dir)
    if image_dir and os.path.isdir(image_dir):
        if not npz_search_dirs or os.path.abspath(cache_dir) != os.path.abspath(image_dir):
            npz_search_dirs.append(image_dir)
    return npz_search_dirs


def _glob_npz(search_dirs: list[str], recursive: bool = False) -> list[str]:
    npz_paths: list[str] = []
    for search_dir in search_dirs:
        if recursive:
            npz_paths.extend(
                glob.glob(os.path.join(search_dir, "**", "*" + CACHE_SUFFIX), recursive=True)
            )
        else:
            npz_paths.extend(glob.glob(os.path.join(search_dir, "*" + CACHE_SUFFIX)))
    return npz_paths


def _build_npz_by_stem(npz_paths: list[str], search_dirs: list[str]) -> dict[str, str]:
    npz_by_stem: dict[str, str] = {}
    for npz_path in npz_paths:
        stem_key = npz_path.rsplit("_", maxsplit=2)[0]
        for root in search_dirs:
            try:
                rel = os.path.relpath(stem_key, root)
                if rel != ".":
                    npz_by_stem.setdefault(rel.replace(os.sep, "/"), npz_path)
                    break
            except ValueError:
                continue
    return npz_by_stem


class TestBuildSearchDirs:
    def test_cache_dir_only(self, tmp_path):
        cache_dir = tmp_path / ".lora"
        cache_dir.mkdir()
        dirs = _build_search_dirs(str(cache_dir), None)
        assert dirs == [str(cache_dir)]

    def test_image_dir_only(self, tmp_path):
        image_dir = tmp_path / ".resized"
        image_dir.mkdir()
        dirs = _build_search_dirs(None, str(image_dir))
        assert dirs == [str(image_dir)]

    def test_both_different_dirs(self, tmp_path):
        cache_dir = tmp_path / ".lora"
        image_dir = tmp_path / ".resized"
        cache_dir.mkdir()
        image_dir.mkdir()
        dirs = _build_search_dirs(str(cache_dir), str(image_dir))
        assert len(dirs) == 2
        assert dirs[0] == str(cache_dir)
        assert dirs[1] == str(image_dir)

    def test_same_dir_no_duplicate(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        dirs = _build_search_dirs(str(shared), str(shared))
        assert len(dirs) == 1
        assert dirs[0] == str(shared)

    def test_cache_dir_nonexistent_skipped(self, tmp_path):
        image_dir = tmp_path / ".resized"
        image_dir.mkdir()
        dirs = _build_search_dirs(str(tmp_path / ".lora"), str(image_dir))
        assert dirs == [str(image_dir)]

    def test_image_dir_nonexistent_skipped(self, tmp_path):
        cache_dir = tmp_path / ".lora"
        cache_dir.mkdir()
        dirs = _build_search_dirs(str(cache_dir), str(tmp_path / ".resized"))
        assert dirs == [str(cache_dir)]

    def test_both_none(self):
        dirs = _build_search_dirs(None, None)
        assert dirs == []


class TestCacheSearchDualDir:
    def test_finds_npz_in_cache_dir(self, tmp_path):
        image_dir = tmp_path / ".resized"
        cache_dir = tmp_path / ".lora"
        image_dir.mkdir()
        cache_dir.mkdir()

        (cache_dir / ("photo_1024x1024" + CACHE_SUFFIX)).touch()

        search_dirs = _build_search_dirs(str(cache_dir), str(image_dir))
        npz_paths = _glob_npz(search_dirs)

        assert len(npz_paths) == 1
        assert npz_paths[0] == str(cache_dir / ("photo_1024x1024" + CACHE_SUFFIX))

    def test_finds_npz_in_image_dir(self, tmp_path):
        image_dir = tmp_path / ".resized"
        cache_dir = tmp_path / ".lora"
        image_dir.mkdir()
        cache_dir.mkdir()

        (image_dir / ("photo_1024x1024" + CACHE_SUFFIX)).touch()

        search_dirs = _build_search_dirs(str(cache_dir), str(image_dir))
        npz_paths = _glob_npz(search_dirs)

        assert len(npz_paths) == 1
        assert npz_paths[0] == str(image_dir / ("photo_1024x1024" + CACHE_SUFFIX))

    def test_cache_dir_takes_priority(self, tmp_path):
        image_dir = tmp_path / ".resized"
        cache_dir = tmp_path / ".lora"
        image_dir.mkdir()
        cache_dir.mkdir()

        (cache_dir / ("photo_1024x1024" + CACHE_SUFFIX)).touch()
        (image_dir / ("photo_1024x1024" + CACHE_SUFFIX)).touch()

        search_dirs = _build_search_dirs(str(cache_dir), str(image_dir))
        npz_paths = _glob_npz(search_dirs)

        assert len(npz_paths) == 2

        npz_by_stem = _build_npz_by_stem(npz_paths, search_dirs)
        matched = npz_by_stem.get("photo")
        assert matched is not None
        assert cache_dir.name in matched

    def test_same_dir_no_duplicate(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()

        (shared / ("photo_1024x1024" + CACHE_SUFFIX)).touch()

        search_dirs = _build_search_dirs(str(shared), str(shared))
        npz_paths = _glob_npz(search_dirs)

        assert len(npz_paths) == 1

    def test_recursive_finds_nested_npz(self, tmp_path):
        image_dir = tmp_path / ".resized"
        cache_dir = tmp_path / ".lora"
        image_dir.mkdir()
        cache_dir.mkdir()
        nested = cache_dir / "sub" / "deep"
        nested.mkdir(parents=True)

        (nested / ("photo_1024x1024" + CACHE_SUFFIX)).touch()

        search_dirs = _build_search_dirs(str(cache_dir), str(image_dir))
        npz_paths = _glob_npz(search_dirs, recursive=True)

        assert len(npz_paths) == 1
        assert "sub" in npz_paths[0] and "deep" in npz_paths[0]

    def test_multiple_files_across_dirs(self, tmp_path):
        image_dir = tmp_path / ".resized"
        cache_dir = tmp_path / ".lora"
        image_dir.mkdir()
        cache_dir.mkdir()

        (cache_dir / ("img_a_1024x1024" + CACHE_SUFFIX)).touch()
        (image_dir / ("img_b_1024x1024" + CACHE_SUFFIX)).touch()
        (image_dir / ("img_c_1024x1024" + CACHE_SUFFIX)).touch()

        search_dirs = _build_search_dirs(str(cache_dir), str(image_dir))
        npz_paths = _glob_npz(search_dirs)

        assert len(npz_paths) == 3
        basenames = {os.path.basename(p) for p in npz_paths}
        assert basenames == {
            "img_a_1024x1024" + CACHE_SUFFIX,
            "img_b_1024x1024" + CACHE_SUFFIX,
            "img_c_1024x1024" + CACHE_SUFFIX,
        }


class TestNpzByStemMapping:
    def test_stem_key_derived_correctly(self, tmp_path):
        cache_dir = tmp_path / ".lora"
        cache_dir.mkdir()

        npz_name = "photo_1024x1024" + CACHE_SUFFIX
        (cache_dir / npz_name).touch()

        search_dirs = _build_search_dirs(str(cache_dir), None)
        npz_paths = _glob_npz(search_dirs)
        npz_by_stem = _build_npz_by_stem(npz_paths, search_dirs)

        assert "photo" in npz_by_stem

    def test_nested_stem_preserves_relative_path(self, tmp_path):
        cache_dir = tmp_path / ".lora"
        sub = cache_dir / "concept_a"
        sub.mkdir(parents=True)

        npz_name = "img_1024x1024" + CACHE_SUFFIX
        (sub / npz_name).touch()

        search_dirs = _build_search_dirs(str(cache_dir), None)
        npz_paths = _glob_npz(search_dirs, recursive=True)
        npz_by_stem = _build_npz_by_stem(npz_paths, search_dirs)

        expected_key = "concept_a/img"
        assert expected_key in npz_by_stem

    def test_setdefault_prefers_first_match(self, tmp_path):
        cache_dir = tmp_path / ".lora"
        image_dir = tmp_path / ".resized"
        cache_dir.mkdir()
        image_dir.mkdir()

        npz_name = "photo_1024x1024" + CACHE_SUFFIX
        cache_npz = cache_dir / npz_name
        image_npz = image_dir / npz_name
        cache_npz.touch()
        image_npz.touch()

        search_dirs = _build_search_dirs(str(cache_dir), str(image_dir))
        npz_paths = _glob_npz(search_dirs)
        npz_by_stem = _build_npz_by_stem(npz_paths, search_dirs)

        assert npz_by_stem["photo"] == str(cache_npz)
