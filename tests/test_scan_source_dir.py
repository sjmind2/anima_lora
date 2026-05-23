import sys
import types
from unittest.mock import MagicMock

import pytest

_pyside6_modules = [
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]


@pytest.fixture(autouse=True)
def _mock_pyside6():
    injected = {}
    for mod_name in _pyside6_modules:
        if mod_name not in sys.modules:
            fake = types.ModuleType(mod_name)
            fake.__path__ = []
            sys.modules[mod_name] = fake
            injected[mod_name] = fake
    if "PySide6.QtCore" not in injected and "PySide6.QtCore" in sys.modules:
        pass
    else:
        qt_core = sys.modules.get("PySide6.QtCore")
        if qt_core is not None and not hasattr(qt_core, "Qt"):
            qt_core.Qt = MagicMock()
    if "PySide6.QtGui" not in injected and "PySide6.QtGui" in sys.modules:
        pass
    else:
        qt_gui = sys.modules.get("PySide6.QtGui")
        if qt_gui is not None and not hasattr(qt_gui, "QPixmap"):
            qt_gui.QPixmap = MagicMock()
    if "PySide6.QtWidgets" not in injected and "PySide6.QtWidgets" in sys.modules:
        pass
    else:
        qt_widgets = sys.modules.get("PySide6.QtWidgets")
        if qt_widgets is not None:
            for name in ("QCheckBox", "QComboBox", "QLabel", "QLineEdit",
                         "QMessageBox", "QSpinBox", "QWidget"):
                if not hasattr(qt_widgets, name):
                    setattr(qt_widgets, name, MagicMock())
    yield
    for mod_name in injected:
        sys.modules.pop(mod_name, None)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class TestScanSourceDir:
    def test_empty_dir(self, tmp_path):
        from gui import scan_source_dir
        result = scan_source_dir(str(tmp_path))
        assert result == []

    def test_nonexistent_dir(self):
        from gui import scan_source_dir
        result = scan_source_dir("/nonexistent/path/xyz")
        assert result == []

    def test_root_images_only(self, tmp_path):
        from gui import scan_source_dir
        (tmp_path / "img1.png").write_bytes(b"fake")
        (tmp_path / "img2.jpg").write_bytes(b"fake")
        result = scan_source_dir(str(tmp_path))
        assert len(result) == 1
        assert result[0]["name"] == "(root)"
        assert result[0]["num_repeats"] == 1
        assert ".resized" in result[0]["image_dir"]
        assert ".lora" in result[0]["cache_dir"]

    def test_subdirs_with_images(self, tmp_path):
        from gui import scan_source_dir
        (tmp_path / "img.png").write_bytes(b"fake")
        (tmp_path / "x").mkdir()
        (tmp_path / "x" / "a.png").write_bytes(b"fake")
        (tmp_path / "y").mkdir()
        (tmp_path / "y" / "b.png").write_bytes(b"fake")
        result = scan_source_dir(str(tmp_path))
        assert len(result) == 3
        assert result[0]["name"] == "(root)"
        names = [r["name"] for r in result]
        assert "x" in names
        assert "y" in names

    def test_num_repeats_extraction(self, tmp_path):
        from gui import scan_source_dir
        (tmp_path / "10_abc").mkdir()
        (tmp_path / "10_abc" / "a.png").write_bytes(b"fake")
        (tmp_path / "xyz").mkdir()
        (tmp_path / "xyz" / "b.png").write_bytes(b"fake")
        (tmp_path / "100_concept").mkdir()
        (tmp_path / "100_concept" / "c.png").write_bytes(b"fake")
        result = scan_source_dir(str(tmp_path))
        by_name = {r["name"]: r for r in result}
        assert by_name["10_abc"]["num_repeats"] == 10
        assert by_name["xyz"]["num_repeats"] == 1
        assert by_name["100_concept"]["num_repeats"] == 100

    def test_hidden_dirs_skipped(self, tmp_path):
        from gui import scan_source_dir
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "a.png").write_bytes(b"fake")
        (tmp_path / "visible").mkdir()
        (tmp_path / "visible" / "b.png").write_bytes(b"fake")
        result = scan_source_dir(str(tmp_path))
        names = [r["name"] for r in result]
        assert ".hidden" not in names
        assert "visible" in names

    def test_path_format(self, tmp_path):
        from gui import scan_source_dir
        basename = tmp_path.name
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.png").write_bytes(b"fake")
        result = scan_source_dir(str(tmp_path))
        sub = result[0]
        assert sub["image_dir"] == f"post_image_dataset/{basename}/sub/.resized"
        assert sub["cache_dir"] == f"post_image_dataset/{basename}/sub/.lora"

    def test_root_first(self, tmp_path):
        from gui import scan_source_dir
        (tmp_path / "img.png").write_bytes(b"fake")
        (tmp_path / "aaa").mkdir()
        (tmp_path / "aaa" / "a.png").write_bytes(b"fake")
        (tmp_path / "zzz").mkdir()
        (tmp_path / "zzz" / "b.png").write_bytes(b"fake")
        result = scan_source_dir(str(tmp_path))
        assert result[0]["name"] == "(root)"


class TestApplyDatasetOverrides:
    def test_subsets_replaced(self):
        from library.config.io import _apply_dataset_overrides
        blueprint = {
            "general": {"keep_tokens": 3},
            "datasets": [
                {
                    "resolution": 1024,
                    "batch_size": 1,
                    "subsets": [{"image_dir": "old", "num_repeats": 1}],
                }
            ],
        }
        override = {
            "datasets": [
                {
                    "subsets": [
                        {"image_dir": "new1", "num_repeats": 5},
                        {"image_dir": "new2", "num_repeats": 10},
                    ]
                }
            ],
        }
        _apply_dataset_overrides(blueprint, override)
        subsets = blueprint["datasets"][0]["subsets"]
        assert len(subsets) == 2
        assert subsets[0]["image_dir"] == "new1"
        assert subsets[1]["image_dir"] == "new2"

    def test_no_subsets_override(self):
        from library.config.io import _apply_dataset_overrides
        blueprint = {
            "general": {"keep_tokens": 3},
            "datasets": [
                {
                    "resolution": 1024,
                    "batch_size": 1,
                    "subsets": [{"image_dir": "old", "num_repeats": 1}],
                }
            ],
        }
        override = {"datasets": [{"batch_size": 2}]}
        _apply_dataset_overrides(blueprint, override)
        assert blueprint["datasets"][0]["batch_size"] == 2
        assert blueprint["datasets"][0]["subsets"] == [{"image_dir": "old", "num_repeats": 1}]
