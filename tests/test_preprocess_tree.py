from PIL import Image
from library.preprocess.images import process_image


def _make_image(path, w=64, h=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (255, 0, 0)).save(path)


BUCKET_ARGS = ((1024, 1024), 512, 2048, 64, True)


class TestResizeTreeMode:
    def test_tree_directory_structure(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_image(src / "root.png")
        (src / "sub").mkdir()
        _make_image(src / "sub" / "child.png")

        root_out = dst / ".resized"
        root_out.mkdir(parents=True)
        sub_out = dst / "sub" / ".resized"
        sub_out.mkdir(parents=True)

        process_image(src / "root.png", root_out, BUCKET_ARGS, copy_captions=False)
        process_image(
            src / "sub" / "child.png", sub_out, BUCKET_ARGS, copy_captions=False
        )

        assert (root_out / "root.png").exists()
        assert (sub_out / "child.png").exists()

    def test_same_name_no_conflict(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        (src / "x").mkdir(parents=True)
        (src / "y").mkdir(parents=True)
        _make_image(src / "x" / "photo.png")
        _make_image(src / "y" / "photo.png")

        x_out = dst / "x" / ".resized"
        y_out = dst / "y" / ".resized"
        x_out.mkdir(parents=True)
        y_out.mkdir(parents=True)

        process_image(src / "x" / "photo.png", x_out, BUCKET_ARGS, copy_captions=False)
        process_image(src / "y" / "photo.png", y_out, BUCKET_ARGS, copy_captions=False)

        assert (x_out / "photo.png").exists()
        assert (y_out / "photo.png").exists()

    def test_output_is_png(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_image(src / "test.jpg")
        out = dst / ".resized"
        out.mkdir(parents=True)

        process_image(src / "test.jpg", out, BUCKET_ARGS, copy_captions=False)

        assert (out / "test.png").exists()

    def test_small_image_still_processed(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_image(src / "tiny.png", w=32, h=32)
        out = dst / ".resized"
        out.mkdir(parents=True)

        process_image(src / "tiny.png", out, BUCKET_ARGS, copy_captions=False)

        assert (out / "tiny.png").exists()
