from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


BIG = (900, 900)
SMALL = (64, 64)


class TestResizeTreeMode:
    def test_tree_creates_per_subset_resized_dirs(self, tmp_path: Path) -> None:
        from library.preprocess import resize_to_buckets

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_image(src / "root.png", BIG)
        _write_image(src / "4_a" / "img.png", BIG)
        _write_image(src / "5_b" / "img.png", BIG)

        stats, _ = resize_to_buckets(src, dst, tree=True, workers=1, verbose=False)

        assert stats.written == 3
        assert (dst / ".resized" / "root.png").exists()
        assert (dst / "4_a" / ".resized" / "img.png").exists()
        assert (dst / "5_b" / ".resized" / "img.png").exists()

    def test_tree_independent_stem_namespaces(self, tmp_path: Path) -> None:
        from library.preprocess import resize_to_buckets

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_image(src / "x" / "photo.png", BIG)
        _write_image(src / "y" / "photo.png", BIG)

        stats, _ = resize_to_buckets(src, dst, tree=True, workers=1, verbose=False)

        assert stats.written == 2
        assert (dst / "x" / ".resized" / "photo.png").exists()
        assert (dst / "y" / ".resized" / "photo.png").exists()

    def test_tree_skips_hidden_dirs(self, tmp_path: Path) -> None:
        from library.preprocess import resize_to_buckets

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_image(src / ".hidden" / "img.png", BIG)
        _write_image(src / "visible" / "img.png", BIG)

        stats, _ = resize_to_buckets(src, dst, tree=True, workers=1, verbose=False)

        assert stats.written == 1
        assert not (dst / ".hidden" / ".resized" / "img.png").exists()
        assert (dst / "visible" / ".resized" / "img.png").exists()

    def test_tree_empty_root_only_subdirs(self, tmp_path: Path) -> None:
        from library.preprocess import resize_to_buckets

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_image(src / "4_a" / "img.png", BIG)

        stats, _ = resize_to_buckets(src, dst, tree=True, workers=1, verbose=False)

        assert stats.written == 1
        assert (dst / "4_a" / ".resized" / "img.png").exists()
        assert not (dst / ".resized").exists()

    def test_tree_min_pixels_filter_per_subset(self, tmp_path: Path) -> None:
        from library.preprocess import resize_to_buckets

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_image(src / "tiny.png", SMALL)
        _write_image(src / "4_a" / "big.png", BIG)

        stats, _ = resize_to_buckets(src, dst, tree=True, workers=1, verbose=False)

        assert stats.written == 1
        assert stats.skipped == 1
        assert not (dst / ".resized" / "tiny.png").exists()
        assert (dst / "4_a" / ".resized" / "big.png").exists()

    def test_tree_false_unchanged_behavior(self, tmp_path: Path) -> None:
        from library.preprocess import resize_to_buckets

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_image(src / "a.png", BIG)

        stats, bucket_counts = resize_to_buckets(
            src, dst, tree=False, workers=1, verbose=False
        )

        assert stats.seen == 1
        assert stats.written == 1
        assert (dst / "a.png").exists()
        assert not (dst / ".resized").exists()

    def test_tree_copies_captions_per_subset(self, tmp_path: Path) -> None:
        from library.preprocess import resize_to_buckets

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_image(src / "root.png", BIG)
        (src / "root.txt").write_text("root caption")
        _write_image(src / "4_a" / "img.png", BIG)
        (src / "4_a" / "img.txt").write_text("subset caption")

        stats, _ = resize_to_buckets(src, dst, tree=True, workers=1, verbose=False)

        assert stats.written == 2
        assert (dst / ".resized" / "root.txt").read_text() == "root caption"
        assert (dst / "4_a" / ".resized" / "img.txt").read_text() == "subset caption"


class TestCacheLatentsTreeMode:
    def test_tree_discovers_subsets_with_resized_dirs(self, tmp_path: Path) -> None:
        from library.preprocess.latents import _cache_latents_tree

        data_dir = tmp_path / "data"
        _write_image(data_dir / "4_a" / ".resized" / "img.png", BIG)

        class _FakeVAE:
            device = None
            dtype = None

            def encode_pixels_to_latents(self, batch):
                import torch

                c = batch.shape[0]
                h, w = batch.shape[-2], batch.shape[-1]
                self.device = batch.device
                self.dtype = batch.dtype
                return torch.randn(c, 16, h // 8, w // 8)

        vae = _FakeVAE()
        stats = _cache_latents_tree(data_dir, vae, batch_size=4)

        assert stats.seen == 1
        assert stats.written == 1
        assert (data_dir / "4_a" / ".lora").is_dir()

    def test_tree_skips_dirs_without_resized(self, tmp_path: Path) -> None:
        from library.preprocess.latents import _cache_latents_tree

        data_dir = tmp_path / "data"
        _write_image(data_dir / "4_a" / ".resized" / "img.png", BIG)
        _write_image(data_dir / "4_b" / "img.png", BIG)

        class _FakeVAE:
            device = None
            dtype = None

            def encode_pixels_to_latents(self, batch):
                import torch

                c = batch.shape[0]
                h, w = batch.shape[-2], batch.shape[-1]
                return torch.randn(c, 16, h // 8, w // 8)

        vae = _FakeVAE()
        stats = _cache_latents_tree(data_dir, vae, batch_size=4)

        assert stats.seen == 1
        assert not (data_dir / "4_b" / ".lora").is_dir()

    def test_tree_processes_root_resized(self, tmp_path: Path) -> None:
        from library.preprocess.latents import _cache_latents_tree

        data_dir = tmp_path / "data"
        _write_image(data_dir / ".resized" / "root.png", BIG)

        class _FakeVAE:
            device = None
            dtype = None

            def encode_pixels_to_latents(self, batch):
                import torch

                c = batch.shape[0]
                h, w = batch.shape[-2], batch.shape[-1]
                return torch.randn(c, 16, h // 8, w // 8)

        vae = _FakeVAE()
        stats = _cache_latents_tree(data_dir, vae, batch_size=4)

        assert stats.seen == 1
        assert stats.written == 1
        assert (data_dir / ".lora").is_dir()

    def test_tree_empty_returns_zero_stats(self, tmp_path: Path) -> None:
        from library.preprocess.latents import _cache_latents_tree

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        class _FakeVAE:
            device = None
            dtype = None

            def encode_pixels_to_latents(self, batch):
                import torch

                return torch.zeros(1)

        vae = _FakeVAE()
        stats = _cache_latents_tree(data_dir, vae, batch_size=4)

        assert stats.seen == 0
        assert stats.written == 0


class TestCliTreeArg:
    def test_include_tree_adds_flag(self) -> None:
        import argparse

        from library.runtime.cli import add_io_args

        parser = argparse.ArgumentParser()
        add_io_args(parser, dir_required=False, include_tree=True)
        args = parser.parse_args(["--tree"])
        assert args.tree is True

    def test_tree_default_false(self) -> None:
        import argparse

        from library.runtime.cli import add_io_args

        parser = argparse.ArgumentParser()
        add_io_args(parser, dir_required=False, include_tree=True)
        args = parser.parse_args([])
        assert args.tree is False

    def test_include_tree_false_no_flag(self) -> None:
        import argparse

        from library.runtime.cli import add_io_args

        parser = argparse.ArgumentParser()
        add_io_args(parser, dir_required=False, include_tree=False)
        with pytest.raises(SystemExit):
            parser.parse_args(["--tree"])
