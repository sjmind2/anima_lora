# Preprocess Bucket Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add detailed resolution descriptions to bucket family options and a dataset distribution analysis button in the workflow frontend preprocess stage.

**Architecture:** Schema-driven approach — extend `preprocess.yaml` with `choice_details` metadata, add a new API endpoint for bucket stats, and enhance `FieldRenderer.js` to render detail lines and an inline stats table. Core scanning logic extracted to `library/datasets/buckets.py` and shared with the GUI.

**Tech Stack:** Python (aiohttp backend), Vue.js (frontend), YAML schema, pytest (tests)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `library/datasets/buckets.py` | Shared `scan_dataset_bucket_distribution()` function |
| `gui/__init__.py` | Refactored to import shared function instead of local implementation |
| `workflow/app.py` | New `POST /api/dataset/bucket-stats` endpoint |
| `workflow/schemas/preprocess.yaml` | Extended with `choice_details` per family |
| `workflow/web/js/api.js` | New `analyzeBucketStats()` method |
| `workflow/web/js/components/FieldRenderer.js` | Enhanced rendering for `list[str]` with `choice_details` |
| `tests/test_bucket_distribution.py` | Unit tests for the shared scanning function |
| `tests/test_workflow_app.py` | Integration test for the new API endpoint |

---

### Task 1: Add shared `scan_dataset_bucket_distribution()` to buckets.py

**Files:**
- Modify: `library/datasets/buckets.py` (append after `scan_images_for_bucket_stats` or after line 89)
- Create: `tests/test_bucket_distribution.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bucket_distribution.py`:

```python
import pytest
from pathlib import Path
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bucket_distribution.py -v`
Expected: FAIL — `ImportError: cannot import name 'scan_dataset_bucket_distribution'`

- [ ] **Step 3: Write the implementation**

Append to `library/datasets/buckets.py` after the `get_bucket_list` function (after line 89):

```python
IMAGE_EXTS_SCAN = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def scan_dataset_bucket_distribution(source_dir: str, enabled_families: list[str]) -> dict:
    src = Path(source_dir)
    if not src.is_dir():
        return {"error": "Directory not found"}

    all_tc = {name: info['tc'] for name, info in BUCKET_FAMILIES.items()}
    enabled_tc = {name: all_tc[name] for name in enabled_families if name in all_tc}

    original_counts = {name: 0 for name in all_tc}
    resized_counts = {name: 0 for name in all_tc}

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
            best_enabled = min(enabled_tc.items(), key=lambda kv: abs(kv[1] * 256 - img_area))[0]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bucket_distribution.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add library/datasets/buckets.py tests/test_bucket_distribution.py
git commit -m "feat: add scan_dataset_bucket_distribution to library/datasets/buckets.py"
```

---

### Task 2: Refactor gui/__init__.py to use shared function

**Files:**
- Modify: `gui/__init__.py:1063-1091`

- [ ] **Step 1: Replace local implementation with import**

Replace the body of `scan_images_for_bucket_stats` in `gui/__init__.py` (lines 1063-1091) with:

```python
def scan_images_for_bucket_stats(source_dir: str, enabled_families: list[str]) -> dict[str, int]:
    from library.datasets.buckets import scan_dataset_bucket_distribution

    result = scan_dataset_bucket_distribution(source_dir, enabled_families)
    if "error" in result:
        return {}
    return {name: info["resized"] for name, info in result["families"].items()}
```

- [ ] **Step 2: Verify GUI import still works**

Run: `.venv\Scripts\python.exe -c "from gui import scan_images_for_bucket_stats; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add gui/__init__.py
git commit -m "refactor: gui uses shared scan_dataset_bucket_distribution"
```

---

### Task 3: Add API endpoint for bucket stats

**Files:**
- Modify: `workflow/app.py` (add route + handler)
- Modify: `tests/test_workflow_app.py` (add integration test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_app.py`:

```python
class TestBucketStatsAPI:
    @pytest.mark.asyncio
    async def test_bucket_stats_missing_dir(self, client):
        resp = await client.post("/api/dataset/bucket-stats", json={
            "source_dir": "/nonexistent/path",
            "enabled_families": ["M", "L"],
        })
        assert resp.status == 200
        data = await resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_bucket_stats_empty_dir(self, client, tmp_path):
        empty = tmp_path / "empty_imgs"
        empty.mkdir()
        resp = await client.post("/api/dataset/bucket-stats", json={
            "source_dir": str(empty),
            "enabled_families": ["M", "L"],
        })
        assert resp.status == 200
        data = await resp.json()
        assert data["total_images"] == 0
        assert "families" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_workflow_app.py::TestBucketStatsAPI -v`
Expected: FAIL — 404 (route not found)

- [ ] **Step 3: Add the route and handler**

In `workflow/app.py`, add the route after line 51 (after the existing routes):

```python
    app.router.add_post("/api/dataset/bucket-stats", _handle_bucket_stats)
```

Add the handler function before `start_server` (before line 307):

```python
async def _handle_bucket_stats(req: web.Request) -> web.Response:
    body = await req.json()
    source_dir = body.get("source_dir", "")
    enabled_families = body.get("enabled_families", [])
    if not source_dir:
        return web.json_response({"error": "source_dir is required"}, status=400)
    from library.datasets.buckets import scan_dataset_bucket_distribution
    result = scan_dataset_bucket_distribution(source_dir, enabled_families)
    return web.json_response(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_workflow_app.py::TestBucketStatsAPI -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add workflow/app.py tests/test_workflow_app.py
git commit -m "feat: add POST /api/dataset/bucket-stats endpoint"
```

---

### Task 4: Extend preprocess.yaml schema with choice_details

**Files:**
- Modify: `workflow/schemas/preprocess.yaml`

- [ ] **Step 1: Update the schema**

Replace the `bucket_families` field definition in `workflow/schemas/preprocess.yaml` (lines 19-33) with:

```yaml
      - key: bucket_families
        type: "list[str]"
        required: true
        label: "分辨率分组"
        choices: ["S1", "S2", "XS", "S", "M", "L", "XL"]
        choice_labels:
          S1: "S1 (TC=1024)"
          S2: "S2 (TC=4096)"
          XS: "XS (TC=1680)"
          S: "S (TC=2160)"
          M: "M (TC=3600)"
          L: "L (TC=4032)"
          XL: "XL (TC=5040)"
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
            resolutions: ["640x2016", "672x1920", "720x1792", "768x1680", "896x1440", "960x1344", "1008x1280", "1120x1152"]
        default: ["S1"]
        help: "选择训练图片的目标分辨率分组。可多选组合。"
```

- [ ] **Step 2: Verify schema loads correctly**

Run: `.venv\Scripts\python.exe -c "from workflow.config import load_schema; s=load_schema('preprocess'); print(s['groups'][1]['fields'][0].get('choice_details',{}).keys())"`
Expected: `dict_keys(['S1', 'S2', 'XS', 'S', 'M', 'L', 'XL'])`

- [ ] **Step 3: Commit**

```bash
git add workflow/schemas/preprocess.yaml
git commit -m "feat: add choice_details with resolutions to preprocess schema"
```

---

### Task 5: Add analyzeBucketStats to api.js

**Files:**
- Modify: `workflow/web/js/api.js`

- [ ] **Step 1: Add the API method**

Add before the closing `return {` block in `workflow/web/js/api.js` (insert after the `connectEventStream` function definition, around line 43), and add the method to the returned object.

In the returned object (after `connectEventStream: connectEventStream,` at line 110), add:

```javascript
    analyzeBucketStats: function(sourceDir, enabledFamilies) {
      return post("/api/dataset/bucket-stats", {
        source_dir: sourceDir,
        enabled_families: enabledFamilies,
      });
    },
```

- [ ] **Step 2: Verify syntax**

Run: `node -c workflow/web/js/api.js`
Expected: No syntax errors

- [ ] **Step 3: Commit**

```bash
git add workflow/web/js/api.js
git commit -m "feat: add analyzeBucketStats to API client"
```

---

### Task 6: Enhance FieldRenderer.js for detail lines + stats button

**Files:**
- Modify: `workflow/web/js/components/FieldRenderer.js`
- Modify: `workflow/web/css/style.css` (add stats-related styles)

This is the most complex task. The `list[str]` template section (lines 214-228) needs to be expanded to support:
1. A detail line below each option showing TC + resolutions (when `choice_details` exists)
2. A stats line showing original/resized counts (populated after analysis)
3. An "分析数据集" button at the top of the option area
4. CSS styles for the new elements

- [ ] **Step 1: Add CSS styles**

Append to `workflow/web/css/style.css`:

```css
.bucket-detail-line {
  font-size: 10px;
  color: var(--text-dim);
  line-height: 1.3;
  margin-top: 1px;
  padding-left: 2px;
}

.bucket-stats-line {
  font-size: 11px;
  margin-top: 1px;
  padding-left: 2px;
}

.bucket-stats-line .stats-original {
  color: var(--text-dim);
}

.bucket-stats-line .stats-resized {
  color: var(--accent);
  font-weight: 500;
}

.bucket-stats-line .stats-resized.dim {
  color: var(--text-dim);
  font-weight: normal;
}

.bucket-analyze-btn {
  font-size: 11px;
  padding: 2px 8px;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  color: var(--text);
  margin-left: auto;
}

.bucket-analyze-btn:hover {
  border-color: var(--accent);
}

.bucket-analyze-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.bucket-options-area {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.bucket-option-item {
  display: flex;
  flex-direction: column;
  min-width: 120px;
  padding: 4px 8px;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}

.bucket-option-item.active {
  border-color: var(--accent);
  background: rgba(60, 120, 200, 0.15);
}

.bucket-option-item input[type="checkbox"] {
  display: none;
}

.bucket-option-header {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
}
```

Also remove the existing `.combo-switch` and `.combo-switch.active` styles if they're only used for bucket_families (lines 882-903), since the new `.bucket-option-item` replaces them. **Check first** — if `.combo-switch` is used elsewhere, keep it.

- [ ] **Step 2: Add data properties and methods to FieldRenderer**

Add to the FieldRenderer component, inside `data` (or computed), the `bucketStats` reactive property and the `analyzeBucketStats` method.

Since FieldRenderer uses plain object (not Vue 3 Composition API), add a `data` function and methods:

In the `methods` section of the FieldRenderer component (after `toggleListItem`), add:

```javascript
      analyzeBucketStats: function () {
        var self = this;
        var allValues = this.allValues || {};
        var sourceDir = allValues.source_image_dir || "";
        if (!sourceDir) return;
        var selected = this.currentValue || [];
        self._bucketStatsLoading = true;
        AnimaAPI.analyzeBucketStats(sourceDir, selected).then(function (result) {
          self._bucketStatsLoading = false;
          if (result.error) {
            self._bucketStatsError = result.error;
            self._bucketStats = null;
          } else {
            self._bucketStats = result;
            self._bucketStatsError = null;
          }
        }).catch(function () {
          self._bucketStatsLoading = false;
          self._bucketStatsError = "Request failed";
          self._bucketStats = null;
        });
      },
```

Initialize `_bucketStats`, `_bucketStatsError`, `_bucketStatsLoading` in the component. Since this is a Vue 2-style options component without a `data` function, use `created` hook or initialize as properties. Add to the component object:

```javascript
    created: function () {
      this._bucketStats = null;
      this._bucketStatsError = null;
      this._bucketStatsLoading = false;
    },
```

- [ ] **Step 3: Replace the list[str] template section**

Replace the `list[str]` template section (lines 214-228) with the enhanced version that handles both regular `list[str]` and bucket-family-aware rendering:

```javascript
      '  <div v-if="field.type === \'list[str]\' && field.choice_details"',
      '    class="bucket-options-area">',
      '    <div v-if="field.help || field.key === \'bucket_families\'" style="display:flex;width:100%;margin-bottom:4px;">',
      '      <span v-if="field.help" style="font-size:11px;color:var(--text-dim);">{{ field.help }}</span>',
      '      <button class="bucket-analyze-btn"',
      '        :disabled="!allValues.source_image_dir"',
      '        @click="analyzeBucketStats">',
      '        {{ _bucketStatsLoading ? "分析中..." : "分析数据集" }}',
      '      </button>',
      '    </div>',
      '    <div v-for="opt in (field.choices || [])" :key="opt"',
      '      class="bucket-option-item"',
      '      :class="{ active: (currentValue || []).includes(opt) }"',
      '      @click="toggleListItem(opt)">',
      '      <input type="checkbox"',
      '        :checked="(currentValue || []).includes(opt)"',
      '        style="display: none;" />',
      '      <div class="bucket-option-header">',
      '        <span>{{ (field.choice_labels || {})[opt] || opt }}</span>',
      '      </div>',
      '      <div v-if="field.choice_details[opt]" class="bucket-detail-line">',
      '        {{ field.choice_details[opt].resolutions.join("  ") }}',
      '      </div>',
      '      <div v-if="_bucketStats && _bucketStats.families[opt]" class="bucket-stats-line">',
      '        <span class="stats-original">原始: {{ _bucketStats.families[opt].original }}张</span>',
      '        <span style="margin: 0 4px;">·</span>',
      '        <span class="stats-resized" :class="{ dim: !_bucketStats.families[opt].resized }">',
      '          缩放后: {{ _bucketStats.families[opt].resized }}张',
      '        </span>',
      '      </div>',
      '    </div>',
      '  </div>',
      '',
      '  <div v-if="field.type === \'list[str]\' && !field.choice_details"',
      '    style="display: flex; flex-wrap: wrap; gap: 6px;">',
      '    <label v-for="opt in (field.choices || [])" :key="opt"',
      '      class="combo-switch"',
      '      :class="{ active: (currentValue || []).includes(opt) }"',
      '      style="cursor: pointer; font-size: 11px;"',
      '      :title="(field.choice_labels || {})[opt] || opt">',
      '      <input type="checkbox"',
      '        :checked="(currentValue || []).includes(opt)"',
      '        @change="toggleListItem(opt)"',
      '        style="display: none;" />',
      '      {{ (field.choice_labels || {})[opt] || opt }}',
      '    </label>',
      '    <div v-if="field.help" style="font-size:11px;color:var(--text-dim);width:100%;margin-top:2px;">{{ field.help }}</div>',
      '  </div>',
```

- [ ] **Step 4: Verify JS syntax**

Run: `node -c workflow/web/js/components/FieldRenderer.js`
Expected: No syntax errors

- [ ] **Step 5: Manual browser test**

Start the workflow server and open the preprocess stage. Verify:
- Each bucket family option shows resolutions in gray text below the label
- The "分析数据集" button appears
- Clicking the button (with a valid source_image_dir) shows stats per family
- Selecting/deselecting options still works correctly

- [ ] **Step 6: Commit**

```bash
git add workflow/web/js/components/FieldRenderer.js workflow/web/css/style.css
git commit -m "feat: render bucket detail lines and stats in FieldRenderer"
```

---

### Task 7: Run full test suite and verify

- [ ] **Step 1: Run all tests**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run linter**

Run: `.venv\Scripts\python.exe -m ruff check . --fix && .venv\Scripts\python.exe -m ruff format .`
Expected: No errors

- [ ] **Step 3: Final commit if any lint fixes**

```bash
git add -A
git commit -m "chore: lint fixes"
```
