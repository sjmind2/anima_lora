# Regularization Dataset Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to specify an independent regularization source directory whose subdirectories become `is_reg=true` subsets alongside training subsets, with unified preprocessing and zero changes to `train.py` or the daemon.

**Architecture:** The GUI gains a `reg_source_dir` input field. When scanned, it calls the existing `scan_source_dir()` to discover subdirectories, marks each subset `is_reg=true`, and merges them into the subset list. The variant TOML stores all subsets (train + reg) in `[[datasets.subsets]]`. Preprocessing already reads subsets from the variant TOML, so reg images are automatically cached. CLI users set `REG_SOURCE_DIR` env var, which triggers the same auto-scan. `train.py` and daemon need no changes.

**Tech Stack:** Python, PySide6, TOML config, existing `scan_source_dir()` / `_auto_scan_subsets()` / `BlueprintGenerator` infrastructure.

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `gui/__init__.py` | Modify `scan_source_dir()` | Add optional `is_reg` param to mark all returned subsets |
| `gui/tabs/config_tab.py` | Modify | Add `reg_source_dir` UI, scan logic, subset card "REG" badge, `prior_loss_weight` control |
| `gui/i18n/en.py` | Modify | Add i18n keys for reg UI |
| `gui/i18n/cn.py` | Modify | Add Chinese translations |
| `gui/i18n/ko.py` | Modify | Add Korean translations |
| `gui/i18n/ja.py` | Modify | Add Japanese translations |
| `scripts/tasks/preprocess.py` | Modify `_auto_scan_subsets()` | Accept optional `reg_source_dir`, scan and append reg subsets |
| `scripts/tasks/_common.py` | Modify `train()` | Read `REG_SOURCE_DIR` env var and pass to preprocessing |

---

### Task 1: Add `is_reg` parameter to `scan_source_dir()`

**Files:**
- Modify: `gui/__init__.py:857-913`

- [ ] **Step 1: Modify `scan_source_dir()` signature and add `is_reg` to each subset dict**

In `gui/__init__.py`, change the function signature at line 857 from:

```python
def scan_source_dir(source_dir: str) -> list[dict]:
```

to:

```python
def scan_source_dir(source_dir: str, *, is_reg: bool = False) -> list[dict]:
```

Then add `"is_reg": is_reg` to both subset dicts:
- The `(root)` subset dict (around line 869-877): add `"is_reg": is_reg,` after `"recursive": True,`
- The child directory subset dict (around line 891-898): add `"is_reg": is_reg,` after `"recursive": True,`

Also update the logger calls to include `is_reg` in the info messages at lines 878-880 and 901-908 by appending `is_reg=is_reg` to the format args.

- [ ] **Step 2: Verify existing callers are unaffected**

`scan_source_dir()` is called from:
- `gui/tabs/config_tab.py:577` — `self._subsets = scan_source_dir(src)` — no `is_reg` arg, defaults to `False` ✓
- `gui/tabs/config_tab.py:840` — `self._subsets = scan_source_dir(src)` — same ✓

No changes needed at call sites in this task.

---

### Task 2: Add `is_reg` parameter to `_auto_scan_subsets()`

**Files:**
- Modify: `scripts/tasks/preprocess.py:508-537`

- [ ] **Step 1: Modify `_auto_scan_subsets()` signature and add `is_reg` to each subset dict**

In `scripts/tasks/preprocess.py`, change line 508 from:

```python
def _auto_scan_subsets(source_dir: str) -> list[dict]:
```

to:

```python
def _auto_scan_subsets(source_dir: str, *, is_reg: bool = False) -> list[dict]:
```

Add `"is_reg": is_reg,` to both subset dicts (the `(root)` dict around line 516-523 and the child dict around line 529-536).

- [ ] **Step 2: Extend `cmd_preprocess()` to scan reg source dir**

In `scripts/tasks/preprocess.py`, modify `cmd_preprocess()` (lines 540-564). After the existing subset loading/scanning block and before the `if subsets:` check at line 558, add reg scanning:

```python
def cmd_preprocess(extra):
    subsets = _load_subset_configs()
    if not subsets:
        source_image_dir = _path("source_image_dir", "image_dataset")
        print(
            f"  cmd_preprocess: no subsets in method TOML — auto-scanning "
            f"source_image_dir={source_image_dir!r}",
            file=sys.stderr,
        )
        subsets = _auto_scan_subsets(source_image_dir)
        if subsets:
            print(f"  cmd_preprocess: auto-scan found {len(subsets)} subset(s)", file=sys.stderr)

    # Append regularization subsets if REG_SOURCE_DIR is set
    reg_dir = os.environ.get("REG_SOURCE_DIR", "").strip()
    if reg_dir and Path(reg_dir).is_dir():
        reg_subsets = _auto_scan_subsets(reg_dir, is_reg=True)
        if reg_subsets:
            print(f"  cmd_preprocess: reg auto-scan found {len(reg_subsets)} subset(s) from {reg_dir!r}", file=sys.stderr)
            subsets = (subsets or []) + reg_subsets

    if subsets:
        print(f"  cmd_preprocess: tree mode with {len(subsets)} subset(s)", file=sys.stderr)
        cmd_preprocess_subsets(extra, subsets=subsets)
        return

    print("  cmd_preprocess: no source data found, aborting", file=sys.stderr)
    sys.exit(1)
```

Note: `import os` is already available at the top of the file.

---

### Task 3: Add i18n keys for regularization UI

**Files:**
- Modify: `gui/i18n/en.py`
- Modify: `gui/i18n/cn.py`
- Modify: `gui/i18n/ko.py`
- Modify: `gui/i18n/ja.py`

- [ ] **Step 1: Add English keys**

In `gui/i18n/en.py`, add these keys to the `STRINGS` dict (before the closing `}`):

```python
    "reg_source_dir": "Regularization Source Dir",
    "reg_scan_subsets": "Scan Reg",
    "reg_scan_subsets_tooltip": "Scan the regularization source directory for subdirectories",
    "reg_subsets_section": "Regularization Subsets",
    "reg_badge": "REG",
    "prior_loss_weight": "Prior Loss Weight",
    "prior_loss_weight_tooltip": "Loss weight for regularization images (1.0 = equal, lower = less influence)",
    "reg_scan_no_dir": "No regularization source directory specified or directory is empty.",
```

- [ ] **Step 2: Add Chinese keys**

In `gui/i18n/cn.py`, add:

```python
    "reg_source_dir": "正则化源目录",
    "reg_scan_subsets": "扫描正则",
    "reg_scan_subsets_tooltip": "扫描正则化源目录的子目录",
    "reg_subsets_section": "正则化子集",
    "reg_badge": "正则",
    "prior_loss_weight": "先验损失权重",
    "prior_loss_weight_tooltip": "正则化图片的损失权重（1.0 = 等权，更低 = 影响更小）",
    "reg_scan_no_dir": "未指定正则化源目录或目录为空。",
```

- [ ] **Step 3: Add Korean keys**

In `gui/i18n/ko.py`, add:

```python
    "reg_source_dir": "정규화 소스 디렉토리",
    "reg_scan_subsets": "정규 스캔",
    "reg_scan_subsets_tooltip": "정규화 소스 디렉토리의 하위 디렉토리를 스캔합니다",
    "reg_subsets_section": "정규화 서브셋",
    "reg_badge": "정규",
    "prior_loss_weight": "사전 손실 가중치",
    "prior_loss_weight_tooltip": "정규화 이미지의 손실 가중치 (1.0 = 동일, 낮을수록 영향 감소)",
    "reg_scan_no_dir": "정규화 소스 디렉토리가 지정되지 않았거나 디렉토리가 비어 있습니다.",
```

- [ ] **Step 4: Add Japanese keys**

In `gui/i18n/ja.py`, add:

```python
    "reg_source_dir": "正則化ソースディレクトリ",
    "reg_scan_subsets": "正則スキャン",
    "reg_scan_subsets_tooltip": "正則化ソースディレクトリのサブディレクトリをスキャン",
    "reg_subsets_section": "正則化サブセット",
    "reg_badge": "正則",
    "prior_loss_weight": "事前損失重み",
    "prior_loss_weight_tooltip": "正則化画像の損失重み（1.0 = 同等、低いほど影響が小さい）",
    "reg_scan_no_dir": "正則化ソースディレクトリが指定されていないか、ディレクトリが空です。",
```

---

### Task 4: Add `reg_source_dir` and `prior_loss_weight` to GUI ConfigTab

**Files:**
- Modify: `gui/tabs/config_tab.py`

This is the largest task. It modifies the config tab to add reg source dir input, reg scanning, reg subset display, and prior_loss_weight control.

- [ ] **Step 1: Add `reg_source_dir` input field with scan button**

In `gui/tabs/config_tab.py`, inside the `_reload()` method's `_build_subgroup_box()` inner function, after the `if k == "source_image_dir":` block (lines 412-428), add a similar block for `reg_source_dir`:

```python
                if k == "source_image_dir":
                    # ... existing code (lines 412-428) ...
                elif k == "reg_source_dir":
                    reg_scan_btn = QPushButton(t("reg_scan_subsets"))
                    reg_scan_btn.setToolTip(t("reg_scan_subsets_tooltip"))
                    reg_scan_btn.clicked.connect(self._scan_reg_subsets)
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.addWidget(w, 1)
                    row_layout.addWidget(reg_scan_btn)
                    form.addRow(lbl, row_widget)
                    self._reg_source_dir_widget = w
                else:
                    form.addRow(lbl, w)
```

This requires changing the existing `else` at line 429 to `elif` and adding the new branch. The original code structure is:

```python
                if k == "source_image_dir":
                    ...
                else:
                    form.addRow(lbl, w)
```

Change to:

```python
                if k == "source_image_dir":
                    ...
                elif k == "reg_source_dir":
                    reg_scan_btn = QPushButton(t("reg_scan_subsets"))
                    reg_scan_btn.setToolTip(t("reg_scan_subsets_tooltip"))
                    reg_scan_btn.clicked.connect(self._scan_reg_subsets)
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.addWidget(w, 1)
                    row_layout.addWidget(reg_scan_btn)
                    form.addRow(lbl, row_widget)
                    self._reg_source_dir_widget = w
                else:
                    form.addRow(lbl, w)
```

- [ ] **Step 2: Add `_scan_reg_subsets` method**

Add after `_scan_subsets` (after line 583):

```python
    def _scan_reg_subsets(self):
        src = self._reg_source_dir_widget.text().strip() if hasattr(self, "_reg_source_dir_widget") else ""
        if not src:
            logger.warning("_scan_reg_subsets: reg_source_dir is empty")
            QMessageBox.information(self, t("reg_subsets_section"), t("reg_scan_no_dir"))
            return
        logger.info("_scan_reg_subsets: scanning reg_source_dir=%r", src)
        reg_subsets = scan_source_dir(src, is_reg=True)
        logger.info("_scan_reg_subsets: scan returned %d reg subset(s)", len(reg_subsets))
        # Remove existing reg subsets, keep train subsets
        self._subsets = [s for s in self._subsets if not s.get("is_reg")]
        self._subsets.extend(reg_subsets)
        self._rebuild_subset_ui()
        if not reg_subsets:
            QMessageBox.information(self, t("reg_subsets_section"), t("reg_scan_no_dir"))
        else:
            self._mark_dirty()
```

- [ ] **Step 3: Show "REG" badge on reg subset cards in `_build_subsets_box()`**

In `_build_subsets_box()` (line 656-700), after the name label is created at line 672-674, add a "REG" badge for reg subsets:

Change lines 672-674 from:

```python
            name_lbl = QLabel(sub["name"])
            name_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
            card_layout.addWidget(name_lbl)
```

to:

```python
            name_row = QHBoxLayout()
            name_lbl = QLabel(sub["name"])
            name_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
            name_row.addWidget(name_lbl)
            if sub.get("is_reg"):
                reg_badge = QLabel(t("reg_badge"))
                reg_badge.setStyleSheet(
                    "background-color: #5b5bd6; color: white; "
                    "border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;"
                )
                name_row.addWidget(reg_badge)
            name_row.addStretch()
            card_layout.addLayout(name_row)
```

- [ ] **Step 4: Add `prior_loss_weight` control**

In `_build_subsets_box()`, after the subsets_box is created and before returning it (between the loop and `subsets_box.setLayout(subsets_layout)`), add a `prior_loss_weight` spin box. Insert after the loop (after line 698) and before `subsets_box.setLayout(subsets_layout)` (line 699):

```python
        # prior_loss_weight control
        has_reg = any(s.get("is_reg") for s in self._subsets)
        if has_reg:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setFrameShadow(QFrame.Sunken)
            subsets_layout.addWidget(sep)
            pw_row = QHBoxLayout()
            pw_label = QLabel(t("prior_loss_weight"))
            pw_label.setToolTip(t("prior_loss_weight_tooltip"))
            pw_row.addWidget(pw_label)
            self._prior_loss_weight_spin = QDoubleSpinBox()
            self._prior_loss_weight_spin.setRange(0.0, 10.0)
            self._prior_loss_weight_spin.setSingleStep(0.1)
            self._prior_loss_weight_spin.setDecimals(2)
            self._prior_loss_weight_spin.setValue(
                self._prior_loss_weight if hasattr(self, "_prior_loss_weight") else 1.0
            )
            self._prior_loss_weight_spin.setFixedWidth(80)
            self._prior_loss_weight_spin.wheelEvent = lambda e: e.ignore()
            self._prior_loss_weight_spin.valueChanged.connect(self._mark_dirty)
            pw_row.addWidget(self._prior_loss_weight_spin)
            pw_row.addStretch()
            subsets_layout.addLayout(pw_row)
```

- [ ] **Step 5: Load `is_reg` and `prior_loss_weight` in `_reload()`**

In the subset loading loop inside `_reload()` (around lines 499-506), add `is_reg` to the entry dict:

Change line 506 from:

```python
                    entry = {
                        "name": name,
                        "source_dir": source_dir,
                        "image_dir": image_dir,
                        "cache_dir": cache_dir,
                        "num_repeats": sub.get("num_repeats", 1),
                        "recursive": sub.get("recursive", True),
                    }
```

to:

```python
                    entry = {
                        "name": name,
                        "source_dir": source_dir,
                        "image_dir": image_dir,
                        "cache_dir": cache_dir,
                        "num_repeats": sub.get("num_repeats", 1),
                        "recursive": sub.get("recursive", True),
                        "is_reg": sub.get("is_reg", False),
                    }
```

Also, after the subset loading block and before `if self._subsets:` (line 517), add loading of `prior_loss_weight` from variant data:

```python
        self._prior_loss_weight = variant_data.get("prior_loss_weight", 1.0)
```

- [ ] **Step 6: Save `prior_loss_weight` in `_save_preset()`**

In `_save_preset()`, after the subset serialization block (after line 874), and before the extra-args textarea section (line 878), add:

```python
        # Save prior_loss_weight if there are reg subsets
        has_reg = any(s.get("is_reg") for s in self._subsets)
        if has_reg and hasattr(self, "_prior_loss_weight_spin"):
            out["prior_loss_weight"] = self._prior_loss_weight_spin.value()
        elif not has_reg:
            out.pop("prior_loss_weight", None)
```

- [ ] **Step 7: Auto-scan reg dir when saving with no reg subsets**

In `_save_preset()`, after the existing auto-scan block for train subsets (lines 836-845), add a similar block for reg subsets:

```python
        # Auto-scan reg_source_dir if no reg subsets exist
        has_reg = any(s.get("is_reg") for s in self._subsets)
        if not has_reg:
            reg_src = self._reg_source_dir_widget.text().strip() if hasattr(self, "_reg_source_dir_widget") else ""
            if reg_src:
                logger.info("_save_preset: no reg subsets — auto-scanning reg_source_dir=%r", reg_src)
                reg_subsets = scan_source_dir(reg_src, is_reg=True)
                if reg_subsets:
                    self._subsets.extend(reg_subsets)
                    self._rebuild_subset_ui()
                    logger.info("_save_preset: reg auto-scan produced %d subset(s)", len(reg_subsets))
```

- [ ] **Step 8: Handle reg subsets in `_on_source_dir_changed`**

When the user changes `source_image_dir`, we should re-scan train subsets but keep reg subsets. Modify `_scan_subsets()` to only replace non-reg subsets:

Change `_scan_subsets()` (lines 570-583) from:

```python
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
```

to:

```python
    def _scan_subsets(self):
        src = self._source_dir_widget.text().strip() if hasattr(self, "_source_dir_widget") else ""
        if not src:
            logger.warning("_scan_subsets: source_image_dir is empty, cannot scan")
            QMessageBox.information(self, t("subsets_section"), t("subsets_scan_no_dir"))
            return
        logger.info("_scan_subsets: scanning source_image_dir=%r", src)
        train_subsets = scan_source_dir(src)
        # Preserve existing reg subsets
        reg_subsets = [s for s in self._subsets if s.get("is_reg")]
        self._subsets = train_subsets + reg_subsets
        logger.info("_scan_subsets: scan returned %d train subset(s), %d reg subset(s) preserved", len(train_subsets), len(reg_subsets))
        self._rebuild_subset_ui()
        if not self._subsets:
            QMessageBox.information(self, t("subsets_section"), t("subsets_scan_no_dir"))
        else:
            self._mark_dirty()
```

---

### Task 5: Add `REG_SOURCE_DIR` env var support to CLI path

**Files:**
- Modify: `scripts/tasks/_common.py`

- [ ] **Step 1: Pass `REG_SOURCE_DIR` to preprocessing env**

In `scripts/tasks/_common.py`, the `_launch_preprocess()` within ConfigTab already passes `METHOD`, `METHODS_SUBDIR`, `PRESET` env vars (see `gui/tabs/config_tab.py:1061-1065`). The preprocessing reads subsets from the variant TOML (which now includes reg subsets from the GUI save). So the GUI→daemon path already works.

For the CLI path (`make lora` → `tasks.py lora`), the preprocessing is called separately via `make preprocess` or `tasks.py preprocess`. The `cmd_preprocess()` in Task 2 already reads `REG_SOURCE_DIR` env var.

For the training path, `train.py` reads the variant TOML which already has `is_reg` subsets serialized. No `_common.py` changes needed for training.

However, to make `REG_SOURCE_DIR` work with the CLI preprocessing + training flow, we need to ensure the env var is available. The user would run:

```bash
REG_SOURCE_DIR=reg_dataset make preprocess
make lora
```

The preprocessing will scan reg dir and write reg subsets into the variant TOML (if using `gui-methods` path). But for the standard `make lora` path (no variant TOML), the preprocessing uses `_auto_scan_subsets` which doesn't write to a TOML — it passes subsets directly to `cmd_preprocess_subsets`.

This means for the CLI path, `REG_SOURCE_DIR` works for preprocessing (reg subsets get cached), but training needs a separate mechanism to load reg subsets. The cleanest approach: add `REG_SOURCE_DIR` handling to `_auto_scan_subsets` in preprocessing (done in Task 2) and document that CLI users should use `--dataset_config` for training with reg data, or use the GUI path which serializes everything to TOML.

No additional code changes needed for this task — Task 2 already covers the preprocessing side.

---

### Task 6: Verification

- [ ] **Step 1: Verify variant TOML serialization**

Run the GUI, set a `reg_source_dir`, click scan reg, save, then inspect the variant TOML file. Confirm:
- `reg_source_dir` is saved as a top-level key
- `prior_loss_weight` is saved if reg subsets exist
- Each reg subset has `is_reg = true`
- Each train subset has `is_reg = false` (or absent)

- [ ] **Step 2: Verify preprocessing includes reg subsets**

Submit a preprocess+train chain from GUI. Check daemon logs that:
- Reg subsets are included in the subset count
- Preprocessing creates cache files in the reg cache directories
- Training log shows reg images counted

- [ ] **Step 3: Verify CLI path with REG_SOURCE_DIR**

Run:
```powershell
$env:REG_SOURCE_DIR = "reg_dataset"
python tasks.py preprocess
```

Confirm stderr output includes reg subset count.

- [ ] **Step 4: Run linter**

```powershell
.venv\Scripts\activate
ruff check gui\__init__.py gui\tabs\config_tab.py scripts\tasks\preprocess.py --fix
ruff format gui\__init__.py gui\tabs\config_tab.py scripts\tasks\preprocess.py
```
