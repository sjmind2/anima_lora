from __future__ import annotations

from PIL import Image

from library.datasets.buckets import scan_dataset_bucket_distribution


class TestScanDatasetBucketDistribution:
    def test_empty_dir_returns_zeros(self, tmp_path):
        result = scan_dataset_bucket_distribution(str(tmp_path), ["M", "L"])
        assert result["total_images"] == 0
        for fam in result["families"].values():
            assert fam["original"] == 0
            assert fam["resized"] == 0

    def test_single_image_matches_nearest_family(self, tmp_path):
        img = Image.new("RGB", (512, 512))
        img.save(tmp_path / "test.png")
        result = scan_dataset_bucket_distribution(str(tmp_path), ["S1"])
        assert result["total_images"] == 1
        assert result["families"]["S1"]["original"] == 1
        assert result["families"]["S1"]["resized"] == 1

    def test_resized_absorbs_unselected_families(self, tmp_path):
        img = Image.new("RGB", (512, 512))
        img.save(tmp_path / "test.png")
        result = scan_dataset_bucket_distribution(str(tmp_path), ["XL"])
        assert result["total_images"] == 1
        assert result["families"]["S1"]["original"] == 1
        assert result["families"]["S1"]["resized"] == 0
        assert result["families"]["XL"]["resized"] == 1

    def test_nonexistent_dir_returns_error(self):
        result = scan_dataset_bucket_distribution("/nonexistent/path", ["M"])
        assert "error" in result

    def test_all_families_original_sum_equals_total(self, tmp_path):
        for i, size in enumerate([(512, 512), (1024, 1024), (640, 672)]):
            img = Image.new("RGB", size)
            img.save(tmp_path / f"img{i}.png")
        result = scan_dataset_bucket_distribution(str(tmp_path), ["M", "L"])
        assert result["total_images"] == 3
        orig_sum = sum(f["original"] for f in result["families"].values())
        assert orig_sum == 3
        resized_sum = sum(f["resized"] for f in result["families"].values())
        assert resized_sum == 3

    def test_empty_enabled_families_uses_all_for_resized(self, tmp_path):
        img = Image.new("RGB", (512, 512))
        img.save(tmp_path / "test.png")
        result = scan_dataset_bucket_distribution(str(tmp_path), [])
        assert result["total_images"] == 1
        assert result["families"]["S1"]["original"] == 1
        assert result["families"]["S1"]["resized"] == 1

    def test_non_image_files_skipped(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.json").write_text("{}")
        result = scan_dataset_bucket_distribution(str(tmp_path), ["S1"])
        assert result["total_images"] == 0

    def test_result_includes_all_families(self, tmp_path):
        result = scan_dataset_bucket_distribution(str(tmp_path), ["M"])
        from library.datasets.buckets import BUCKET_FAMILIES

        assert set(result["families"].keys()) == set(BUCKET_FAMILIES.keys())

    def test_multiple_images_distribute_correctly(self, tmp_path):
        from library.datasets.buckets import BUCKET_FAMILIES

        s1_area = BUCKET_FAMILIES["S1"]["tc"] * 256
        m_area = BUCKET_FAMILIES["M"]["tc"] * 256
        side_s1 = int(s1_area**0.5)
        side_m = int(m_area**0.5)
        for i in range(3):
            Image.new("RGB", (side_s1, side_s1)).save(tmp_path / f"small{i}.png")
        for i in range(2):
            Image.new("RGB", (side_m, side_m)).save(tmp_path / f"med{i}.png")
        result = scan_dataset_bucket_distribution(str(tmp_path), ["S1", "M"])
        assert result["total_images"] == 5
        assert result["families"]["S1"]["original"] == 3
        assert result["families"]["M"]["original"] == 2
        assert result["families"]["S1"]["resized"] == 3
        assert result["families"]["M"]["resized"] == 2
