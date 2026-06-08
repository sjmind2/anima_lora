# Shared Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared cache option to preprocess stages so that identical configurations reuse cached outputs across runs within the same workflow.

**Architecture:** Preprocess stages gain a `shared_cache` boolean (default true). When enabled, the scheduler routes the stage's working directory to `{wf_dir}/shared_cache/{stage_id}/` instead of `{run_dir}/{stage_id}/`. The executor computes a SHA256 hash of the resolved config and skips preprocessing if the hash matches the previous run. Train stages are unchanged — they follow whatever path the preprocess output provides.

**Tech Stack:** Python, Pydantic, TOML (tomli_w), SHA256 via hashlib/json canonical serialization

---

### Task 1: Add `shared_cache` field to preprocess schema

**Files:**
- Modify: `workflow/schemas/preprocess.yaml`

- [ ] **Step 1: Add `shared_cache` field to preprocess schema**

Add a new top-level group `cache` as the first group in the schema, before `advanced`.

```yaml
# workflow/schemas/preprocess.yaml
type: preprocess
label: "预处理"
description: "图像缩放 + VAE缓存 + 文本嵌入缓存"

groups:
  - name: cache
    label: "缓存"
    fields:
      - key: shared_cache
        type: bool
        required: false
        default: true
        label: "共享缓存"

  - name: advanced
    label: "高级覆盖"
    collapsed: true
    fields:
      # ... existing fields unchanged ...
```

The full file becomes:

```yaml
type: preprocess
label: "预处理"
description: "图像缩放 + VAE缓存 + 文本嵌入缓存"

groups:
  - name: cache
    label: "缓存"
    fields:
      - key: shared_cache
        type: bool
        required: false
        default: true
        label: "共享缓存"

  - name: advanced
    label: "高级覆盖"
    collapsed: true
    fields:
      - key: pretrained_model_name_or_path
        type: path
        required: false
        label: "DiT 模型覆盖"
        help: "留空使用全局设置"
      - key: qwen3
        type: path
        required: false
        label: "文本编码器覆盖"
        help: "留空使用全局设置"
      - key: vae
        type: path
        required: false
        label: "VAE 模型覆盖"
        help: "留空使用全局设置"

  - name: data_source
    label: "数据源"
    fields:
      - key: source_image_dir
        type: path
        required: true
        label: "原始数据集路径"
        help: "包含原始图片和 caption 文件的目录"
        widget: directory

  - name: bucket
    label: "分辨率设置"
    fields:
      - key: bucket_families
        type: "list[str]"
        required: true
        label: "分辨率分组"
        choices: ["S1", "S2", "XS", "S", "M", "L", "XL"]
        choice_labels:
          S1: "S1 — TC1024 — 256×1024, 512×512"
          S2: "S2 — TC4096 — 512×2048, 1024×1024"
          XS: "XS — TC1680 — 336~640 竖/横"
          S: "S — TC2160 — 384~720 竖/横"
          M: "M — TC3600 — 480~960 竖/横"
          L: "L — TC4032 — 512~1008 竖/横"
          XL: "XL — TC5040 — 640~1120 竖/横"
        choice_details:
          S1:
            tc: 1024
            resolutions: ["256x1024", "512x512"]
          S2:
            tc: 4096
            resolutions: ["512x2048", "1024x1024"]
          XS:
            tc: 1680
            resolutions: ["336x1280", "384x1120", "448x960", "480x896", "560x768", "640x672"]
          S:
            tc: 2160
            resolutions: ["384x1440", "432x1280", "480x1152", "576x960", "640x864", "720x768"]
          M:
            tc: 3600
            resolutions: ["480x1920", "576x1600", "640x1440", "720x1280", "768x1200", "800x1152", "960x960"]
          L:
            tc: 4032
            resolutions: ["512x2016", "576x1792", "672x1536", "768x1344", "896x1152", "1008x1024"]
          XL:
            tc: 5040
            resolutions: ["640x2016", "672x1920", "720x1792", "768x1680", "896x1440", "960x1340", "1008x1280", "1120x1152"]
        default: ["S1"]
        help: "选择训练图片的目标分辨率分组。S1/S2 为固定比例，XS~XL 为自然分布。可多选组合。"

  - name: filter
    label: "过滤选项"
    fields:
      - key: drop_lowres_images
        type: bool
        required: false
        default: true
        label: "过滤低分辨率图片"
      - key: min_pixels
        type: int
        required: false
        default: 500000
        label: "最小像素数"
        condition: "drop_lowres_images == true"
```

- [ ] **Step 2: Commit**

```bash
git add workflow/schemas/preprocess.yaml
git commit -m "feat(workflow): add shared_cache field to preprocess schema"
```

---

### Task 2: Add i18n labels for `shared_cache`

**Files:**
- Modify: `workflow/i18n/locales/zh-CN.json`
- Modify: `workflow/i18n/locales/en.json`
- Modify: `workflow/i18n/locales/ja.json`

- [ ] **Step 1: Add i18n entries to zh-CN.json**

In `schema.preprocess`, add the `cache` group label and `shared_cache` field/help entries. Also add backend log messages.

Under `"schema"."preprocess"."group"`, add `"cache"` key:
```json
"group": { "cache": "缓存", "advanced": "高级覆盖", "data_source": "数据源", "bucket": "分辨率设置", "filter": "过滤选项" }
```

Under `"schema"."preprocess"."field"`, add:
```json
"shared_cache": "共享缓存"
```

Under `"schema"."preprocess"."help"`, add:
```json
"shared_cache": "开启时，同一工作流内相同配置的预处理结果会被复用，避免重复计算"
```

Under `"backend"."stages"`, add:
```json
"sharedCacheHit": "共享缓存命中，跳过预处理 ({stage_id})",
"sharedCacheMismatch": "共享缓存配置不匹配，重新计算预处理 ({stage_id})"
```

- [ ] **Step 2: Add i18n entries to en.json**

Same structure, English values:

Under `"schema"."preprocess"."group"`:
```json
"group": { "cache": "Cache", "advanced": "Advanced Overrides", "data_source": "Data Source", "bucket": "Resolution Settings", "filter": "Filter Options" }
```

Under `"schema"."preprocess"."field"`:
```json
"shared_cache": "Shared Cache"
```

Under `"schema"."preprocess"."help"`:
```json
"shared_cache": "When enabled, identical preprocessing results are reused within the same workflow to avoid redundant computation"
```

Under `"backend"."stages"`:
```json
"sharedCacheHit": "Shared cache hit, skipping preprocess ({stage_id})",
"sharedCacheMismatch": "Shared cache config mismatch, re-running preprocess ({stage_id})"
```

- [ ] **Step 3: Add i18n entries to ja.json**

Same structure, Japanese values:

Under `"schema"."preprocess"."group"`:
```json
"group": { "cache": "キャッシュ", "advanced": "高度な上書き", "data_source": "データソース", "bucket": "解像度設定", "filter": "フィルターオプション" }
```

Under `"schema"."preprocess"."field"`:
```json
"shared_cache": "共有キャッシュ"
```

Under `"schema"."preprocess"."help"`:
```json
"shared_cache": "有効にすると、同じワークフロー内で同じ設定の前処理結果が再利用され、重複計算を回避します"
```

Under `"backend"."stages"`:
```json
"sharedCacheHit": "共有キャッシュヒット、前処理をスキップ ({stage_id})",
"sharedCacheMismatch": "共有キャッシュ設定が不一致、前処理を再実行 ({stage_id})"
```

- [ ] **Step 4: Commit**

```bash
git add workflow/i18n/locales/zh-CN.json workflow/i18n/locales/en.json workflow/i18n/locales/ja.json
git commit -m "feat(workflow): add i18n labels for shared_cache"
```

---

### Task 3: Modify scheduler to route preprocess stage_dir based on shared_cache

**Files:**
- Modify: `workflow/scheduler.py:89-97`

- [ ] **Step 1: Update `_make_executor` in scheduler**

Replace the current `_make_executor` method (lines 89-97):

```python
def _make_executor(self, stage: WorkflowStage, config: dict, run_dir: Path):
    stage_dir = run_dir / stage.id
    stage_dir.mkdir(parents=True, exist_ok=True)
    infra = self.wf.infrastructure or {}
    if stage.type == "preprocess":
        return PreprocessExecutor(stage.id, config, stage_dir, infra)
    elif stage.type == "train":
        return TrainExecutor(stage.id, config, stage_dir, infra)
    raise ValueError(t("backend.scheduler.unknownStageType", type=stage.type))
```

With:

```python
def _make_executor(self, stage: WorkflowStage, config: dict, run_dir: Path):
    infra = self.wf.infrastructure or {}
    if stage.type == "preprocess":
        use_shared = config.get("shared_cache", True)
        if use_shared:
            stage_dir = self.wf_dir / "shared_cache" / stage.id
        else:
            stage_dir = run_dir / stage.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        return PreprocessExecutor(stage.id, config, stage_dir, infra)
    elif stage.type == "train":
        stage_dir = run_dir / stage.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        return TrainExecutor(stage.id, config, stage_dir, infra)
    raise ValueError(t("backend.scheduler.unknownStageType", type=stage.type))
```

Key change: preprocess stages with `shared_cache=True` get `stage_dir` pointing to `wf_dir/shared_cache/{stage_id}/` instead of `run_dir/{stage_id}/`.

- [ ] **Step 2: Commit**

```bash
git add workflow/scheduler.py
git commit -m "feat(workflow): route preprocess stage_dir to shared_cache when enabled"
```

---

### Task 4: Add shared cache logic to PreprocessExecutor

**Files:**
- Modify: `workflow/stages/preprocess.py`

- [ ] **Step 1: Add imports**

Add at the top of the file, after existing imports:

```python
import hashlib
import json
import shutil
```

- [ ] **Step 2: Add `_compute_config_hash` static method**

Add to `PreprocessExecutor` class, after `_resolve_default_model`:

```python
@staticmethod
def _compute_config_hash(config: dict) -> str:
    """Deterministic hash of config contents, independent of key order."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

- [ ] **Step 3: Add `_should_skip_shared_cache` method**

Add to `PreprocessExecutor` class:

```python
def _should_skip_shared_cache(self, resolved_config: dict, on_stdout) -> bool:
    """Check shared cache: return True if preprocessing can be skipped."""
    post_dir = self.stage_dir / "post_image_dataset"
    hash_file = self.stage_dir / ".config_hash"

    if not post_dir.exists():
        return False

    current_hash = self._compute_config_hash(resolved_config)

    if hash_file.exists():
        saved_hash = hash_file.read_text(encoding="utf-8").strip()
        if saved_hash == current_hash:
            if on_stdout:
                from workflow.i18n import t
                on_stdout(
                    self.stage_id,
                    t("backend.stages.sharedCacheHit", stage_id=self.stage_id),
                )
            return True
        else:
            if on_stdout:
                from workflow.i18n import t
                on_stdout(
                    self.stage_id,
                    t("backend.stages.sharedCacheMismatch", stage_id=self.stage_id),
                )
            shutil.rmtree(str(post_dir), ignore_errors=True)

    return False
```

- [ ] **Step 4: Add `_save_config_snapshot` method**

Add to `PreprocessExecutor` class:

```python
def _save_config_snapshot(self, resolved_config: dict) -> None:
    """Save config.toml and .config_hash for shared cache comparison."""
    from workflow.config import save_stage_toml

    save_stage_toml(resolved_config, self.stage_dir / "config.toml")
    hash_val = self._compute_config_hash(resolved_config)
    (self.stage_dir / ".config_hash").write_text(hash_val, encoding="utf-8")
```

- [ ] **Step 5: Modify `execute` method**

Replace the entire `execute` method (lines 136-189) with:

```python
def execute(
    self,
    on_stdout: Callable | None = None,
    on_progress: Callable | None = None,
    stage_outputs: dict | None = None,
) -> StageResult:
    try:
        use_shared = self.config.get("shared_cache", True)
        resolved_config = self.prepare_config(stage_outputs or {})

        # Shared cache skip check
        if use_shared and self._should_skip_shared_cache(resolved_config, on_stdout):
            subsets = self.discover_subsets()
            dataset_dir = str(self.stage_dir / "post_image_dataset")
            outputs: dict[str, Any] = {"dataset_dir": dataset_dir}
            families = self.config.get("bucket_families")
            if families:
                outputs["bucket_families"] = families
            return StageResult(
                success=True,
                outputs=outputs,
                subsets=subsets,
            )

        # Run preprocessing pipeline
        for step_name, cmd_builder in [
            ("resize", self._build_resize_cmd),
            ("vae", self._build_vae_cmd),
            ("te", self._build_te_cmd),
        ]:
            cmd = cmd_builder()
            if on_stdout:
                on_stdout(self.stage_id, f"[COMMAND:{step_name}] " + " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            self._current_proc = proc
            for line in proc.stdout:
                if self.should_stop:
                    proc.terminate()
                    break
                if on_stdout:
                    on_stdout(self.stage_id, line.rstrip())
            proc.wait()
            self._current_proc = None

            if self.should_stop:
                return StageResult(success=False, error="stopped")

            if proc.returncode != 0:
                return StageResult(
                    success=False,
                    error=f"{step_name} failed with exit code {proc.returncode}",
                )

        # Save config snapshot for shared cache
        if use_shared:
            self._save_config_snapshot(resolved_config)

        subsets = self.discover_subsets()
        dataset_dir = str(self.stage_dir / "post_image_dataset")
        outputs = {"dataset_dir": dataset_dir}
        families = self.config.get("bucket_families")
        if families:
            outputs["bucket_families"] = families
        return StageResult(
            success=True,
            outputs=outputs,
            subsets=subsets,
        )
    except Exception as e:
        return StageResult(success=False, error=str(e))
```

Key changes from the original:
1. Computes `resolved_config` at the start via `prepare_config()`
2. Checks shared cache skip before running any subprocess
3. On cache hit: runs `discover_subsets()` on existing dir and returns immediately
4. On cache miss: runs pipeline as before, then saves config snapshot + hash
5. Non-shared (`shared_cache=False`): no hash check, no snapshot save — behaves exactly as before

- [ ] **Step 6: Commit**

```bash
git add workflow/stages/preprocess.py
git commit -m "feat(workflow): add shared cache skip/save logic to PreprocessExecutor"
```

---

### Task 5: Verify the implementation

- [ ] **Step 1: Run existing unit tests**

```bash
cd o:\loratool\anima_lora_fork && .venv\Scripts\python.exe -m pytest tests/ -v --timeout=30
```

Expected: All existing tests pass. No regressions.

- [ ] **Step 2: Verify schema loads correctly**

```bash
cd o:\loratool\anima_lora_fork && .venv\Scripts\python.exe -c "from workflow.config import load_stage_toml; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Verify i18n loads correctly**

```bash
cd o:\loratool\anima_lora_fork && .venv\Scripts\python.exe -c "from workflow.i18n import t; print(t('backend.stages.sharedCacheHit', stage_id='test'))"
```

Expected: Prints the localized shared cache hit message.

- [ ] **Step 4: Verify config hash is deterministic**

```bash
cd o:\loratool\anima_lora_fork && .venv\Scripts\python.exe -c "
from workflow.stages.preprocess import PreprocessExecutor
d1 = {'a': 1, 'b': 2, 'c': {'x': 10, 'y': 20}}
d2 = {'c': {'y': 20, 'x': 10}, 'a': 1, 'b': 2}
h1 = PreprocessExecutor._compute_config_hash(d1)
h2 = PreprocessExecutor._compute_config_hash(d2)
assert h1 == h2, f'Hash mismatch: {h1} != {h2}'
print(f'Hash OK: {h1[:16]}...')
"
```

Expected: `Hash OK: ...` with no assertion error.

- [ ] **Step 5: Commit verification**

No code changes in this task. If all verifications pass, implementation is complete.
