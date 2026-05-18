"""ConfigTab — training config editor with field tooltips and LoRA variant guide."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any

import html

logger = logging.getLogger(__name__)

import toml
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import (
    CONFIGS_DIR,
    IMAGE_EXTS,
    ROOT,
    _GROUPS,
    _K2G,
    _SKIP,
    _VIRTUAL_KEYS,
    _load,
    _read,
    _save,
    _widget,
    apply_validation_choice,
    confirm_existing_caches,
    confirm_resumable_checkpoint,
    is_basic_field,
    list_gui_variants,
    list_methods,
    merged_gui_variant_preset,
    scan_source_dir,
    variant_path,
)
from gui.explanations import field_help, method_guide
from gui.i18n import t
from gui.process import kill_process_tree, make_subprocess_env, setup_kill_safe
from gui.progress import TQDM_RE, TqdmProgressTracker, make_progress_bar


class ClickableLabel(QLabel):
    """QLabel that emits `clicked` on left-click."""

    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class ConfigTab(QWidget):
    def __init__(self, methods: list[str] | None = None):
        super().__init__()
        self._w: dict[str, QWidget] = {}
        self._preprocessed = (ROOT / "post_image_dataset").exists()
        # Advanced section starts collapsed; user's expand/collapse state
        # persists across _reload (variant switches, save round-trips).
        self._advanced_expanded = False
        # Dirty = form has edits not yet flushed to the variant file.
        # Train/Preprocess auto-saves before launching, since the subprocess
        # re-reads the file from disk and would otherwise miss form edits.
        self._dirty = False
        self._subsets: list[dict] = []
        self._subset_widgets: list[dict] = []
        lay = QVBoxLayout(self)

        # Top bar: method + save + preprocess + train + stop
        # The preset combo is intentionally absent — gui-methods variants
        # (lora-8gb, tlora, etc.) already encode the hardware/perf knobs
        # users used to pick via presets, and all saves now write directly to
        # the current variant file (no preset/variant routing distinction).
        # `methods=` lets callers restrict the picker (e.g. the standard tab
        # shows only lora; the experimental tab mounts a method picker for
        # postfix). When only one method is allowed, the picker hides itself.
        top = QHBoxLayout()
        method_items = methods if methods is not None else list_methods()
        self._method_label = QLabel("Method")
        top.addWidget(self._method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(method_items)
        # Size to the longest entry so names like "easycontrol" / "hydralora"
        # don't get visually clipped on first show. setMinimumContentsLength
        # reserves char-width room; the AdjustToContents policy keeps the
        # combo from shrinking back below that on re-layout.
        self.method_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.method_combo.setMinimumContentsLength(
            max((len(m) for m in method_items), default=10)
        )
        self.method_combo.currentTextChanged.connect(
            lambda _: self._on_method_changed()
        )
        top.addWidget(self.method_combo)
        if len(method_items) <= 1:
            self._method_label.setVisible(False)
            self.method_combo.setVisible(False)

        # Variant picker sits inline next to the method picker — selecting a
        # variant swaps the gui-methods/<variant>.toml file the form is bound
        # to. "+ New" creates a custom variant under gui-methods/custom/;
        # "Guide" replays the method-level help in the right panel.
        self._variant_label = QLabel(t("variant"))
        top.addWidget(self._variant_label)
        self.variant_combo = QComboBox()
        # Reserve room for the longest variant stem we ship (e.g.
        # "tlora_ortho_reft", "hydralora-8gb", "custom/<name>"). Without
        # this, Qt sizes to the shortest entry and the displayed text on
        # selection ends up elided with "…".
        self.variant_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.variant_combo.setMinimumContentsLength(20)
        self.variant_combo.currentTextChanged.connect(lambda _: self._reload())
        top.addWidget(self.variant_combo, 1)
        self.new_variant_btn = QPushButton(t("new_variant"))
        self.new_variant_btn.setToolTip(t("new_variant_tooltip"))
        self.new_variant_btn.clicked.connect(self._create_variant)
        top.addWidget(self.new_variant_btn)
        # The right-hand panel already shows the variant guide by default
        # (and on every variant switch via _reload → _show_explain_placeholder).
        # A dedicated "Guide" button next to Save was visually redundant — to
        # return to the guide after clicking a field, just switch variants or
        # click another field.

        self._save_btn = QPushButton(t("save"))
        self._save_btn_idle_style = ""
        self._save_btn_dirty_style = (
            "background:#e67e22;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._save_btn.clicked.connect(self._save_preset)
        top.addWidget(self._save_btn)

        self.preprocess_btn = QPushButton(t("preprocess"))
        self._preprocess_idle_style = (
            "background:#2980b9;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._preprocess_busy_style = (
            "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.preprocess_btn.setStyleSheet(self._preprocess_idle_style)
        self.preprocess_btn.clicked.connect(self._start_preprocess)
        top.addWidget(self.preprocess_btn)

        self.train_btn = QPushButton(t("train"))
        self._train_idle_style = (
            "background:#27ae60;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._train_busy_style = (
            "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.clicked.connect(self._start_training)
        self.train_btn.setEnabled(self._preprocessed)
        top.addWidget(self.train_btn)

        self.test_btn = QPushButton(t("test"))
        self._test_idle_style = (
            "background:#8e44ad;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._test_busy_style = (
            "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.test_btn.setStyleSheet(self._test_idle_style)
        self.test_btn.clicked.connect(self._start_test)
        self.test_btn.setEnabled(self._has_lora_output())
        top.addWidget(self.test_btn)

        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)

        lay.addLayout(top)

        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        lay.addWidget(self.progress)

        # Vertical splitter: config form on top, log on bottom
        vsplit = QSplitter(Qt.Vertical)

        # Horizontal splitter: form on left, explanation panel on right
        hsplit = QSplitter(Qt.Horizontal)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._form = QWidget()
        outer = QVBoxLayout(self._form)
        outer.setContentsMargins(0, 0, 0, 0)

        # Inner container holds the dynamically-rebuilt grouped form fields
        # (cleared on every _reload). The extra-args button and textarea sit
        # below it inside the same scroll area, but outside the cleared layout
        # so they persist across reloads.
        self._form_inner = QWidget()
        self._fl = QVBoxLayout(self._form_inner)
        self._fl.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._form_inner)

        self.extra_args_btn = QPushButton(t("extra_args_toggle"))
        self.extra_args_btn.setCheckable(True)
        self.extra_args_btn.setToolTip(t("extra_args_tooltip"))
        self.extra_args_btn.clicked.connect(self._toggle_extra_args)
        outer.addWidget(self.extra_args_btn)
        self.extra_args_edit = QPlainTextEdit()
        self.extra_args_edit.setPlaceholderText(t("extra_args_placeholder"))
        self.extra_args_edit.setToolTip(t("extra_args_tooltip"))
        self.extra_args_edit.setMaximumHeight(120)
        self.extra_args_edit.setVisible(False)
        self.extra_args_edit.textChanged.connect(self._mark_dirty)
        outer.addWidget(self.extra_args_edit)
        outer.addStretch()

        sc.setWidget(self._form)
        hsplit.addWidget(sc)

        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            "QTextBrowser { font-size: 13px; padding: 12px; background: #2b2b2b; color: #e0e0e0; }"
        )
        self._explain.setMinimumWidth(320)
        self._show_explain_placeholder()
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([720, 420])

        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("log_placeholder"))
        vsplit.addWidget(self.log)

        vsplit.setSizes([500, 200])
        lay.addWidget(vsplit)

        # QProcess for training. The launchers we spawn (``accelerate launch``,
        # ``python tasks.py …``) fork the real training process, which is what
        # holds VRAM. Run the child in its own session so kill_process_tree
        # can take down the whole subtree on Stop / window close.
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)
        self._stdout_buf = ""
        self._stderr_buf = ""

        self._origin: dict[str, str] = {}
        self._reload()

    # Preset selection is no longer surfaced in the GUI — variants encode the
    # hardware/perf knobs that used to live in presets. The merge still uses
    # 'default' under the hood so the form shows reasonable effective values
    # when a variant file is sparse. All saves write to the variant file.
    _IMPLICIT_PRESET = "default"

    def _current_variant(self) -> str:
        """gui-methods variant for the selected method. Falls back to the
        method name itself when no variants are registered (ip_adapter,
        easycontrol)."""
        v = self.variant_combo.currentText()
        return v or self.method_combo.currentText()

    def _on_method_changed(self):
        self._reload()

    def _refresh_variant_row(self, method: str) -> None:
        variants = list_gui_variants(method)
        current = [
            self.variant_combo.itemText(i) for i in range(self.variant_combo.count())
        ]
        # Rebuilding the combo resets currentText to the first item, which
        # would clobber the user's selection on every _reload. Only rebuild
        # when the variant list actually changed (i.e. method family switched).
        if current != variants:
            self.variant_combo.blockSignals(True)
            self.variant_combo.clear()
            if variants:
                self.variant_combo.addItems(variants)
            self.variant_combo.blockSignals(False)

    def _reload(self):
        method = self.method_combo.currentText()
        if not method:
            return
        self._refresh_variant_row(method)
        variant = self._current_variant()
        merged, origin = merged_gui_variant_preset(variant, self._IMPLICIT_PRESET)
        cfg = {k: v for k, v in merged.items() if k not in _SKIP}

        self._origin = origin

        if hasattr(self, "_explain"):
            self._show_explain_placeholder()

        self._w.clear()
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # Partition fields by Basic vs Advanced first, then by sub-group
        # (Architecture/Training/Performance/Paths/Other). Basic stays
        # always-visible; Advanced is wrapped in a collapsible container.
        basic: dict[str, dict] = {g: {} for g in _GROUPS}
        basic["Other"] = {}
        advanced: dict[str, dict] = {g: {} for g in _GROUPS}
        advanced["Other"] = {}
        for k, v in cfg.items():
            sub = _K2G.get(k, "Other")
            (basic if is_basic_field(k) else advanced)[sub][k] = v

        # "preset" origin shows where the value comes from today, but on Save
        # everything routes to the variant file — no preset/variant split.
        variant_label = f"gui-methods/{variant}.toml"
        origin_style = {
            "base": (
                "color:#888; text-decoration: underline dotted;",
                "from base.toml",
            ),
            "preset": (
                "color:#6aa4d8; text-decoration: underline dotted;",
                f"from presets.toml[{self._IMPLICIT_PRESET}] (saves to {variant_label})",
            ),
            "method": (
                "color:#f0f0f0; text-decoration: underline dotted;",
                f"from {variant_label}",
            ),
        }

        def _build_subgroup_box(gn: str, flds: dict) -> QGroupBox:
            box = QGroupBox(gn)
            form = QFormLayout()
            for k in sorted(flds):
                w = _widget(flds[k], key=k)
                self._w[k] = w
                lbl = ClickableLabel(k)

                help_text = field_help(k)
                style, note = origin_style.get(
                    self._origin.get(k, "base"), origin_style["base"]
                )
                lbl.setStyleSheet(style)
                notes = (note,)

                lbl.clicked.connect(
                    lambda _k=k, _h=help_text, _n=notes: self._show_explain(_k, _h, _n)
                )

                if k == "source_image_dir":
                    scan_btn = QPushButton(t("scan_subsets"))
                    scan_btn.setToolTip(t("scan_subsets_tooltip"))
                    scan_btn.clicked.connect(self._scan_subsets)
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.addWidget(w, 1)
                    row_layout.addWidget(scan_btn)
                    form.addRow(lbl, row_widget)
                    self._source_dir_widget = w
                    w.editingFinished.connect(self._on_source_dir_changed)
                else:
                    form.addRow(lbl, w)
            box.setLayout(form)
            return box

        # Basic — flat list of sub-group boxes.
        basic_box = QGroupBox(t("basic_section"))
        basic_layout = QVBoxLayout()
        basic_layout.setContentsMargins(8, 12, 8, 8)
        for gn, flds in basic.items():
            if not flds:
                continue
            basic_layout.addWidget(_build_subgroup_box(gn, flds))
        basic_box.setLayout(basic_layout)
        self._fl.addWidget(basic_box)

        # Advanced — collapsible. QGroupBox.setCheckable + a child container
        # whose visibility is bound to the checkbox gives a free toggle UI.
        advanced_box = QGroupBox(t("advanced_section"))
        advanced_box.setCheckable(True)
        advanced_box.setChecked(self._advanced_expanded)
        adv_outer = QVBoxLayout()
        adv_outer.setContentsMargins(8, 12, 8, 8)
        adv_inner = QWidget()
        adv_inner_layout = QVBoxLayout(adv_inner)
        adv_inner_layout.setContentsMargins(0, 0, 0, 0)
        for gn, flds in advanced.items():
            if not flds:
                continue
            adv_inner_layout.addWidget(_build_subgroup_box(gn, flds))
        adv_inner.setVisible(self._advanced_expanded)
        adv_outer.addWidget(adv_inner)
        advanced_box.setLayout(adv_outer)

        def _on_advanced_toggled(checked: bool, _inner=adv_inner):
            self._advanced_expanded = checked
            _inner.setVisible(checked)

        advanced_box.toggled.connect(_on_advanced_toggled)
        self._fl.addWidget(advanced_box)

        self._subsets = []
        self._subset_widgets = []
        variant_data = _load(variant_path(variant))
        existing_datasets = variant_data.get("datasets")
        if isinstance(existing_datasets, list) and existing_datasets:
            first_ds = existing_datasets[0]
            if isinstance(first_ds, dict) and "subsets" in first_ds:
                logger.info(
                    "_reload: variant %r has %d subset(s) in [[datasets]], loading into UI",
                    variant, len(first_ds["subsets"]),
                )
                src_dir_base = variant_data.get("source_image_dir", "")
                for idx, sub in enumerate(first_ds["subsets"]):
                    image_dir = sub.get("image_dir", "")
                    cache_dir = sub.get("cache_dir", "")
                    name = sub.get("name", "")
                    source_dir = sub.get("source_dir", "")
                    if not name and image_dir:
                        p = Path(image_dir)
                        if p.name == ".resized":
                            parent_name = p.parent.name
                            name = "(root)" if not parent_name or parent_name == p.parent.parent.name else parent_name
                        else:
                            name = p.name
                    if not source_dir and src_dir_base:
                        if name == "(root)":
                            source_dir = src_dir_base
                        else:
                            source_dir = str(Path(src_dir_base) / name)
                    entry = {
                        "name": name,
                        "source_dir": source_dir,
                        "image_dir": image_dir,
                        "cache_dir": cache_dir,
                        "num_repeats": sub.get("num_repeats", 1),
                        "recursive": sub.get("recursive", True),
                    }
                    self._subsets.append(entry)
                    logger.debug(
                        "_reload: loaded subset[%d] name=%r  image_dir=%r  cache_dir=%r  num_repeats=%d",
                        idx, entry["name"], entry["image_dir"], entry["cache_dir"], entry["num_repeats"],
                    )
            else:
                logger.debug("_reload: variant %r has no subsets in [[datasets]]", variant)
        else:
            logger.debug("_reload: variant %r has no [[datasets]] section", variant)

        if self._subsets:
            subsets_box = self._build_subsets_box()
            self._fl.addWidget(subsets_box)

        self._fl.addStretch()

        # Reload rebuilt the form to match disk → no pending edits.
        # Connect change signals AFTER the values have been seeded by _widget,
        # so the initial setValue/addItems calls don't trip the dirty flag.
        for w in self._w.values():
            self._connect_dirty_signal(w)
        self._clear_dirty()

    # ── Dirty tracking ──

    def _connect_dirty_signal(self, w: QWidget) -> None:
        """Wire each form widget's change signal to _mark_dirty so the Save
        button reflects whether the form has drifted from the variant file."""
        from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox

        if isinstance(w, QComboBox):
            w.currentTextChanged.connect(self._mark_dirty)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(self._mark_dirty)
        elif isinstance(w, QSpinBox):
            w.valueChanged.connect(self._mark_dirty)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(self._mark_dirty)

    def _mark_dirty(self, *_):
        if self._dirty:
            return
        self._dirty = True
        self._update_save_button()

    def _clear_dirty(self):
        self._dirty = False
        self._update_save_button()

    def _update_save_button(self):
        if not hasattr(self, "_save_btn"):
            return
        if self._dirty:
            self._save_btn.setText(t("save") + " *")
            self._save_btn.setStyleSheet(self._save_btn_dirty_style)
            self._save_btn.setToolTip(t("save_dirty_tooltip"))
        else:
            self._save_btn.setText(t("save"))
            self._save_btn.setStyleSheet(self._save_btn_idle_style)
            self._save_btn.setToolTip("")

    # ── Subset scanning ──

    def _scan_subsets(self):
        src = self._source_dir_widget.text().strip() if hasattr(self, "_source_dir_widget") else ""
        if not src:
            logger.warning("_scan_subsets: source_image_dir is empty, cannot scan")
            QMessageBox.information(self, t("subsets_section"), t("subsets_scan_no_dir"))
            return
        logger.info("_scan_subsets: scanning source_image_dir=%r", src)
        self._subsets = scan_source_dir(src)
        logger.info("_scan_subsets: scan returned %d subset(s)", len(self._subsets))
        self._rebuild_subset_ui()
        if not self._subsets:
            QMessageBox.information(self, t("subsets_section"), t("subsets_scan_no_dir"))
        else:
            self._mark_dirty()

    def _on_source_dir_changed(self):
        if self._subsets:
            logger.info("_on_source_dir_changed: source_image_dir changed, re-scanning subsets")
            self._scan_subsets()

    def _build_subsets_box(self) -> QGroupBox:
        subsets_box = QGroupBox(t("subsets_section"))
        subsets_layout = QVBoxLayout()
        subsets_layout.setContentsMargins(8, 12, 8, 8)
        subsets_layout.setSpacing(6)
        self._subset_widgets = []
        for i, sub in enumerate(self._subsets):
            card = QGroupBox()
            card.setStyleSheet(
                "QGroupBox { border: 1px solid #444; border-radius: 4px; "
                "margin-top: 0; padding: 6px; padding-top: 2px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            )
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(4, 4, 4, 4)
            card_layout.setSpacing(2)
            name_lbl = QLabel(sub["name"])
            name_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
            card_layout.addWidget(name_lbl)
            img_lbl = QLabel(sub["image_dir"])
            img_lbl.setStyleSheet("color:#aaa; font-size: 11px;")
            img_lbl.setWordWrap(True)
            img_lbl.setToolTip(sub["image_dir"])
            card_layout.addWidget(img_lbl)
            cache_lbl = QLabel(sub["cache_dir"])
            cache_lbl.setStyleSheet("color:#aaa; font-size: 11px;")
            cache_lbl.setWordWrap(True)
            cache_lbl.setToolTip(sub["cache_dir"])
            card_layout.addWidget(cache_lbl)
            repeats_row = QHBoxLayout()
            repeats_row.addWidget(QLabel(t("subsets_col_num_repeats")))
            repeats_spin = QSpinBox()
            repeats_spin.setRange(1, 1000)
            repeats_spin.setValue(sub["num_repeats"])
            repeats_spin.setFixedWidth(70)
            repeats_spin.wheelEvent = lambda e: e.ignore()
            repeats_spin.valueChanged.connect(self._mark_dirty)
            repeats_row.addWidget(repeats_spin)
            repeats_row.addStretch()
            card_layout.addLayout(repeats_row)
            card.setLayout(card_layout)
            subsets_layout.addWidget(card)
            self._subset_widgets.append({"spin": repeats_spin, "index": i})
        subsets_box.setLayout(subsets_layout)
        return subsets_box

    def _rebuild_subset_ui(self):
        for i in range(self._fl.count()):
            item = self._fl.itemAt(i)
            if item.widget() and isinstance(item.widget(), QGroupBox):
                if item.widget().title() == t("subsets_section"):
                    item.widget().deleteLater()
                    self._fl.removeItem(item)
                    break
        if not self._subsets:
            return
        subsets_box = self._build_subsets_box()
        self._fl.insertWidget(self._fl.count() - 1, subsets_box)

    # ── Explanation panel ──

    def _show_explain_placeholder(self) -> None:
        # When the current method ships variant presets, the right-panel
        # default is the variant guide + Apply-semantics callout (replacing
        # the old collapsible box on the left-side form).
        method = (
            self.method_combo.currentText() if hasattr(self, "method_combo") else ""
        )
        guide = method_guide(method)
        if guide:
            self._explain.setHtml(guide)
            return
        self._explain.setHtml(
            f"<p style='color:#888; font-style:italic;'>{html.escape(t('click_field_for_help'))}</p>"
        )

    def _show_test_output(self) -> None:
        d = ROOT / "output" / "tests"
        imgs: list = []
        if d.is_dir():
            imgs = sorted(
                (p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:4]
        title = html.escape(t("test_output_title"))
        if not imgs:
            self._explain.setHtml(
                f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{title}</h2>"
                f"<p style='color:#888; font-style:italic;'>{html.escape(t('test_output_empty'))}</p>"
            )
            return
        parts = [f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{title}</h2>"]
        for p in imgs:
            url = p.resolve().as_uri()
            parts.append(
                f"<p style='margin:0 0 10px 0;'>"
                f"<img src='{url}' style='max-width:100%;'/><br/>"
                f"<span style='color:#aaa; font-size:11px;'>{html.escape(p.name)}</span>"
                f"</p>"
            )
        self._explain.setHtml("".join(parts))

    def _show_explain(
        self, field: str, help_text: str | None, notes: tuple[str, ...]
    ) -> None:
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{html.escape(field)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:14px; line-height:1.6;'>{html.escape(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:#888; font-style:italic;'>{html.escape(t('no_help_available'))}</p>"
            )
        for note in notes:
            parts.append(
                f"<p style='color:#aaa; font-style:italic; margin-top:12px;'>• {html.escape(note)}</p>"
            )
        self._explain.setHtml("".join(parts))

    # ── Save ──

    def _save_preset(self, *, silent: bool = False):
        """Write the form (and any extra-args TOML) into the current variant
        file. No preset/variant routing — the variant file is the single
        source of truth for the GUI."""
        variant = self._current_variant()
        path = variant_path(variant)

        method_orig = _load(path)
        base = _load(CONFIGS_DIR / "base.toml")
        # Default-preset overlay is the implicit baseline used by _reload, so
        # we treat it as part of the "effective baseline" when deciding which
        # form values are worth writing to disk (skips redundant entries).
        from gui import _load_all_presets  # local import: only needed for save

        implicit_pset = _load_all_presets().get(self._IMPLICIT_PRESET, {})

        out: dict[str, Any] = dict(method_orig)

        for k, w in self._w.items():
            if k in _VIRTUAL_KEYS:
                # Virtual keys (e.g. use_valid) aren't real flat TOML keys —
                # their writeback is handled below via per-key apply helpers.
                continue
            baseline = method_orig.get(k, implicit_pset.get(k, base.get(k)))
            v = _read(w, baseline)
            if k in method_orig or v != baseline:
                out[k] = v

        use_valid_w = self._w.get("use_valid")
        if use_valid_w is not None:
            apply_validation_choice(out, bool(_read(use_valid_w)))

        if self._subsets:
            logger.info("_save_preset: writing %d subset(s) to variant TOML", len(self._subsets))
            dataset_entry = out.get("datasets")
            if not isinstance(dataset_entry, list):
                dataset_entry = [{}]
                out["datasets"] = dataset_entry
            if not dataset_entry:
                dataset_entry.append({})
            first = dataset_entry[0]
            if not isinstance(first, dict):
                first = {}
                dataset_entry[0] = first
            _TRAINING_SUBSET_KEYS = {"image_dir", "cache_dir", "num_repeats", "recursive", "is_reg", "class_tokens", "caption_extension", "keep_tokens", "alpha_mask"}
            subsets_list = []
            for i, sub in enumerate(self._subsets):
                sub_copy = {k: v for k, v in sub.items() if k in _TRAINING_SUBSET_KEYS}
                if self._subset_widgets:
                    for sw in self._subset_widgets:
                        if sw["index"] == i:
                            sub_copy["num_repeats"] = sw["spin"].value()
                            break
                subsets_list.append(sub_copy)
                logger.debug(
                    "_save_preset: subset[%d] name=%r  image_dir=%r  cache_dir=%r  num_repeats=%d  recursive=%s",
                    i, sub.get("name"), sub_copy.get("image_dir"), sub_copy.get("cache_dir"),
                    sub_copy.get("num_repeats"), sub_copy.get("recursive"),
                )
            first["subsets"] = subsets_list
        else:
            logger.debug("_save_preset: no subsets to write")

        # Extra-args textarea: parse as TOML and merge in. Textarea overrides
        # the form for any duplicate key (it's the more explicit signal).
        # Bare backslashes (Windows path paste) break TOML escape parsing —
        # try once verbatim, then retry after \→/ before surfacing the error.
        extra_text = self.extra_args_edit.toPlainText().strip()
        extras: dict[str, Any] = {}
        if extra_text:
            try:
                parsed = toml.loads(extra_text)
            except toml.TomlDecodeError as e:
                if "\\" in extra_text:
                    try:
                        parsed = toml.loads(extra_text.replace("\\", "/"))
                    except toml.TomlDecodeError:
                        QMessageBox.warning(self, t("invalid_toml"), str(e))
                        return
                else:
                    QMessageBox.warning(self, t("invalid_toml"), str(e))
                    return
            extras = {k: v for k, v in parsed.items() if not isinstance(v, dict)}
            out.update(extras)

        path.parent.mkdir(parents=True, exist_ok=True)
        _save(path, out)

        if extras:
            self.extra_args_edit.clear()
            self._reload()  # _reload calls _clear_dirty itself
        else:
            self._clear_dirty()
        if not silent:
            try:
                rel = path.relative_to(CONFIGS_DIR.parent)
            except ValueError:
                rel = path
            QMessageBox.information(self, t("saved"), f"Saved {rel}")

    def _create_variant(self):
        name, ok = QInputDialog.getText(self, t("new_variant"), t("new_variant_prompt"))
        if not ok:
            return
        name = (name or "").strip()
        if not name or not re.match(r"^[A-Za-z0-9_\-]+$", name):
            QMessageBox.warning(self, t("error"), t("new_variant_invalid"))
            return
        full = f"custom/{name}"
        new_path = variant_path(full)
        if new_path.exists():
            QMessageBox.warning(self, t("error"), t("new_variant_exists", name=name))
            return
        new_path.parent.mkdir(parents=True, exist_ok=True)
        # Seed from the currently-selected variant so the form has all
        # method-specific knobs. network_dim / network_alpha only live in
        # gui-methods/<variant>.toml — an empty seed silently drops them from
        # the form, then sparse-diff Save persists nothing, then training
        # falls back to argparse defaults (network_alpha=1) and produces a
        # near-zero-scale adapter. Strip [variant] since it described the
        # source family.
        seed: dict[str, Any] = {}
        current = self.variant_combo.currentText()
        if current:
            seed_path = variant_path(current)
            if seed_path.is_file():
                seed = _load(seed_path)
                seed.pop("variant", None)
        if seed:
            _save(new_path, seed)
        else:
            new_path.write_text("", encoding="utf-8")
        # Rebuild combo and select the new entry. _reload fires via the
        # currentTextChanged signal once we set the index.
        method = self.method_combo.currentText()
        variants = list_gui_variants(method)
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        self.variant_combo.addItems(variants)
        self.variant_combo.blockSignals(False)
        idx = self.variant_combo.findText(full)
        if idx >= 0:
            self.variant_combo.setCurrentIndex(idx)
        else:
            self._reload()

    def _toggle_extra_args(self):
        self.extra_args_edit.setVisible(self.extra_args_btn.isChecked())

    # ── Training ──

    def _has_lora_output(self) -> bool:
        out = ROOT / "output" / "ckpt"
        return out.is_dir() and any(out.glob("*.safetensors"))

    def _start_test(self):
        if not self._has_lora_output():
            QMessageBox.warning(self, t("error"), t("no_lora_for_test"))
            return

        python = sys.executable
        args = ["tasks.py", "test"]

        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "test"
        self._proc.start(python, args)
        self.test_btn.setText(t("test") + " ...")
        self.test_btn.setStyleSheet(self._test_busy_style)
        self.test_btn.setEnabled(False)
        self.preprocess_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)

    def _start_preprocess(self):
        # Flush form edits to disk first — the subprocess re-reads the variant
        # file, so unsaved knobs (paths, source dirs, …) would otherwise be lost.
        if self._dirty:
            self._save_preset(silent=True)

        # Resolve the cache dir that tasks.py preprocess will write into and
        # confirm with the user that any pre-existing caches there will be
        # reused, not wiped. Mirrors scripts/tasks/preprocess.py's fallback.
        variant = self._current_variant()
        merged, _ = merged_gui_variant_preset(variant, self._IMPLICIT_PRESET)
        cache_rel = merged.get("lora_cache_dir") or "post_image_dataset/lora"
        cache_dir = Path(cache_rel)
        if not cache_dir.is_absolute():
            cache_dir = ROOT / cache_dir
        if not confirm_existing_caches(self, cache_dir):
            return

        python = sys.executable
        args = ["tasks.py", "preprocess"]

        # Point tasks.py at the same variant training will use, so any
        # source_image_dir / resized_image_dir / lora_cache_dir override the
        # user wrote into the variant file is honored by preprocess too.
        self._proc.setProcessEnvironment(
            make_subprocess_env(
                METHOD=variant,
                METHODS_SUBDIR="gui-methods",
                PRESET=self._IMPLICIT_PRESET,
            )
        )

        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(
            f"> METHOD={variant} METHODS_SUBDIR=gui-methods python {' '.join(args)}\n"
        )
        self._running_mode = "preprocess"
        self._proc.start(python, args)
        self.preprocess_btn.setText(t("preprocess") + " ...")
        self.preprocess_btn.setStyleSheet(self._preprocess_busy_style)
        self.preprocess_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)

    def _start_training(self):
        if not self._preprocessed:
            QMessageBox.warning(self, t("error"), t("preprocess_required"))
            return

        # Flush form edits to disk first — train.py reads the variant file
        # from disk, so unsaved form values would otherwise be ignored.
        if self._dirty:
            self._save_preset(silent=True)

        variant = self._current_variant()
        merged, _ = merged_gui_variant_preset(variant, self._IMPLICIT_PRESET)
        if not confirm_resumable_checkpoint(self, merged):
            return

        # Flip button visuals to busy + repaint BEFORE the slow accelerate
        # import and QProcess.start, otherwise Qt's event loop is blocked
        # long enough for Windows to flag the GUI as "Not Responding".
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(self._train_busy_style)
        self.train_btn.setEnabled(False)
        self.preprocess_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)
        self.log.clear()
        self._reset_progress()
        # Indeterminate "busy" bar bridges the gap until the child's first
        # tqdm line — without it, Windows reads the gray button + still GUI
        # as "Not Responding" during the multi-second torch import.
        self._progress_tracker.mark_starting(t("starting"))
        QApplication.processEvents()

        # find_spec, not import: actually importing accelerate transitively
        # imports torch, which blocks the GUI thread for several seconds on
        # Windows and freezes the marquee. find_spec just resolves the module
        # location without executing it.
        if importlib.util.find_spec("accelerate.commands.accelerate_cli") is None:
            QMessageBox.warning(self, t("error"), t("accelerate_not_found"))
            self._restore_train_idle()
            return

        # Route through tasks.py rather than spawning accelerate directly:
        # tasks.py uses python.exe + CREATE_NO_WINDOW for its subprocess calls,
        # which keeps tqdm output flowing back to the GUI. If we spawned
        # accelerate from this process (sys.executable = pythonw.exe under the
        # desktop shortcut), accelerate's workers would inherit pythonw and
        # their stdio would silently drop.
        args = ["tasks.py", "lora-gui", variant]

        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "train"
        self._proc.start(sys.executable, args)
        self.stop_btn.setEnabled(True)

    def _restore_train_idle(self):
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.setEnabled(self._preprocessed)
        self.preprocess_btn.setEnabled(True)
        self.test_btn.setEnabled(self._has_lora_output())
        self.method_combo.setEnabled(True)
        self.variant_combo.setEnabled(True)
        self.new_variant_btn.setEnabled(True)

    def _stop_training(self):
        kill_process_tree(self._proc)

    def cleanup_subprocess(self):
        """Hook for app shutdown — kill any running launcher + descendants."""
        kill_process_tree(self._proc)

    def _read_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode(errors="replace")
        self._stdout_buf = self._handle_stream(self._stdout_buf + data)

    def _read_stderr(self):
        data = self._proc.readAllStandardError().data().decode(errors="replace")
        self._stderr_buf = self._handle_stream(self._stderr_buf + data)

    def _handle_stream(self, buf: str) -> str:
        # Split on \n and \r so tqdm carriage-return updates work too.
        parts = re.split(r"[\r\n]", buf)
        tail = parts[-1]  # incomplete trailing fragment — keep buffered
        for line in parts[:-1]:
            if self._progress_tracker.feed(line):
                continue
            if line:
                self._log(line + "\n")
        return tail

    def _reset_progress(self):
        self._stdout_buf = ""
        self._stderr_buf = ""
        self._progress_tracker.reset()

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus):
        # Flush any buffered partial lines before the finish banner.
        for buf_name in ("_stdout_buf", "_stderr_buf"):
            leftover = getattr(self, buf_name, "")
            if leftover and not TQDM_RE.search(leftover):
                self._log(leftover + "\n")
            setattr(self, buf_name, "")
        self.progress.setVisible(False)
        self._log(f"\n{t('finished', code=exit_code)}\n")
        mode = getattr(self, "_running_mode", "train")
        if mode == "preprocess" and exit_code == 0:
            self._preprocessed = True
        if mode == "test" and exit_code == 0:
            self._show_test_output()
        self.preprocess_btn.setText(t("preprocess"))
        self.preprocess_btn.setStyleSheet(self._preprocess_idle_style)
        self.preprocess_btn.setEnabled(True)
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.setEnabled(self._preprocessed)
        self.test_btn.setText(t("test"))
        self.test_btn.setStyleSheet(self._test_idle_style)
        self.test_btn.setEnabled(self._has_lora_output())
        self.stop_btn.setEnabled(False)
        self.method_combo.setEnabled(True)
        self.variant_combo.setEnabled(True)
        self.new_variant_btn.setEnabled(True)

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
