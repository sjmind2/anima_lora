# Workflow Engine — Automated Multi-Stage Training

The Workflow engine is a WebUI + CLI automated training pipeline built on aiohttp (backend) and Vue 3 CDN (frontend). It supports configurable multi-stage training workflows with schema-driven dynamic forms, real-time progress feedback via SSE, and cross-stage checkpoint continuation.

## Installation

### Python Dependencies

All Python dependencies are included in `pyproject.toml`. Install with:

```bash
uv sync
```

Key dependencies:
- `aiohttp >= 3.13.5` — HTTP server and REST API
- `pywebview >= 5.0` — Desktop window mode (optional, falls back to browser)

### Node.js (Optional — Development Only)

> **Not required for normal use.** The Workflow frontend uses Vue 3 via CDN — all JavaScript is pre-bundled in `workflow/web/vendor/`. Node.js is only needed if you want to modify the frontend source files and use a local dev server.

Install Node.js from [nodejs.org](https://nodejs.org/) (LTS recommended) or via package manager:

```bash
# Windows (winget)
winget install OpenJS.NodeJS.LTS

# macOS (Homebrew)
brew install node

# Linux (nvm - recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install --lts
```

### pywebview System Dependency

On Windows, pywebview requires **Microsoft Edge WebView2 Runtime**, which is pre-installed on Windows 10 (1903+) and Windows 11. If missing, download it from [Microsoft](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).

On Linux, pywebview requires `python3-gi` or `python3-pyqt5` — see [pywebview docs](https://pywebview.flowrl.com/guide/installation.html).

### Launching

```bash
# Desktop window mode (default)
python -m workflow

# Browser mode (no pywebview needed)
python -m workflow --no-gui

# Custom port and workflow root
python -m workflow --port 8765 --workflows-root /path/to/workflows
```

## Quick Start: Single-Stage Training

This example walks through creating a basic one-stage LoKR training workflow.

### 1. Launch the Workflow UI

```bash
python -m workflow
```

A desktop window opens at `http://localhost:8765`.

### 2. Create a New Workflow

Click **"New Workflow"** and give it a name (e.g., `my_first_training`).

### 3. Add a Preprocess Stage

1. Click **"Add Stage"** → select **Preprocess**
2. Set **Source directory** to your `image_dataset/` folder
3. Select **Bucket family** — start with `L` (1.03 MP, good balance of quality and speed)
4. Leave **Min pixels** at the default (500,000)

The preprocess stage will resize your images to fit the selected bucket family, then cache VAE latents and text embeddings.

### 4. Add a Train Stage

1. Click **"Add Stage"** → select **Train**
2. Select **Method** — e.g., **LoKR**
3. Configure parameters in the schema-driven form (network_dim, learning_rate, max_train_epochs, etc.)
4. The **Dataset** field automatically references the upstream Preprocess stage's output

### 5. Run

Click **"Run"**. The workflow executes stages in order:

1. **Preprocess** — resizes images, caches VAE latents and text embeddings
2. **Train** — trains the LoKR adapter

### 6. Find Your Training Artifacts

Training outputs are organized under the workflow directory:

```
.anima_workflow/my_first_training/
  runs/
    20260530-120000/          ← timestamped run directory
      preprocess_1/
        post_image_dataset/   ← resized images and caches
      train_1/
        output/
          *.safetensors       ← your trained adapter
        command.txt           ← exact command that was run
        config.toml           ← resolved config
      status.json             ← run status snapshot
      run.log                 ← full log
    latest → 20260530-120000/ ← junction link to latest run
```

**Three ways to find your latest adapter:**

1. **`runs/latest/train_1/output/`** — the `latest` junction always points to the most recent run
2. **History tab** — click the "Open directory" button on any completed run
3. **System log** — shows the safetensors path when training completes

## Single-Stage Usage in Detail

### Preprocess Stage

| Setting | Description |
|---------|-------------|
| **Source directory** | Path to your raw training images (with `.txt` caption sidecars) |
| **Bucket families** | Which resolution family to use. See [Bucket Families guide](bucket-families.md) for details. |
| **Min pixels** | Images below this pixel count are skipped (default: 500,000) |

The preprocess stage runs three sub-steps in order:
1. **Resize** — scales and crops images to fit the selected bucket family
2. **VAE cache** — encodes images to latent space
3. **TE cache** — encodes text captions to embeddings

### Train Stage

The train stage presents a schema-driven form that changes based on the selected method:

- **Method selector** — dropdown to switch between LoRA, LoKR, LoHA, etc.
- **Common parameters** — learning rate, epochs, batch size, optimizer
- **Method-specific parameters** — e.g., `lokr_factor` for LoKR, `network_dim` for LoRA

The form is generated from `workflow/schemas/train_{method}.yaml` and `workflow/schemas/train_common.yaml`.

## Multi-Stage Usage

Multi-stage workflows enable advanced training strategies like [low-resolution pre-training followed by high-resolution refinement](bucket-families.md#multi-stage-training-strategy).

### How Multi-Stage Orchestration Works

Each stage's outputs are available to subsequent stages via:
- **Automatic references** — the system auto-fills `network_weights` and `datasets` from upstream outputs
- **Placeholder syntax** — `${stage_id.output_key}` in config values, resolved at runtime

### Multiple Preprocess Stages

Each Preprocess stage can use different settings:

| Setting | Preprocess 1 | Preprocess 2 |
|---------|-------------|-------------|
| **Bucket families** | `S1` (low resolution, 0.26 MP) | `L` (high resolution, 1.03 MP) |
| **Source directory** | `image_dataset/` | `image_dataset/` (same or different) |

This produces two sets of cached data at different resolutions, each in its own subdirectory.

### Multiple Train Stages

#### `stop_epoch` — Interrupt and Save

Set `stop_epoch` on a Train stage to stop training at a specific epoch and ensure a checkpoint is saved:

```
stop_epoch = 6
```

This sets `max_train_epochs` and `save_every_n_epochs` to the specified value, so training stops immediately after saving the epoch-6 checkpoint.

#### Checkpoint Continuation

When a Train stage runs after another Train stage, it automatically:

1. Finds the upstream stage's `safetensors_path` output
2. Sets `--network_weights` to that path
3. For LoRA: sets `--dim_from_weights` to auto-infer rank from the checkpoint
4. For LyCORIS (lokr/loha/locon): sets `dim_from_weights = false` (dimensions must match config)

#### Typical Multi-Stage Flow

```
Preprocess S1 → Train S1 (stop at epoch 6) → Preprocess L → Train L (from S1 checkpoint)
```

1. **Preprocess S1**: Resize + cache at S1 family (0.26 MP)
2. **Train S1**: Train LoKR adapter, stop at epoch 6
3. **Preprocess L**: Resize + cache at L family (1.03 MP)
4. **Train L**: Continue from S1's epoch-6 checkpoint, using both S1 and L caches

The second Train stage references the first's output via placeholder: `${train_1.safetensors_path}` → resolved to the actual path.

## Log Viewer

The bottom panel has three tabs:

### System Log

Shows workflow-level events: stage start/end, checkpoint saves, errors. Updated in real-time via SSE (Server-Sent Events).

### Script Output

Shows subprocess stdout with:
- **TQDM progress bars** — parsed and displayed as visual progress bars with step count, elapsed time, ETA, and metrics (loss, lr)
- **Stage filtering** — filter output by stage using the dropdown
- **Auto-scroll** — automatically scrolls to latest output; pause/resume with the scroll lock button
- **Buffer limit** — 500 lines per stage; oldest lines are trimmed when exceeded

### Run History

Lists all previous runs in reverse chronological order. Each entry shows:
- **Timestamp** and **duration**
- **Status**: ok / stopped / error / running
- **Stage chain** with color-coded status indicators
- **Actions**: "View log" and "Open directory"

**To find your latest training artifact from history:**
1. Open the **History** tab
2. The most recent run is at the top
3. Click **"Open directory"** to open the run folder
4. Navigate to `{train_stage_id}/output/` to find the `.safetensors` file

Alternatively, `runs/latest` is always a junction/symlink to the most recent run directory.

## Settings

### Language

The UI automatically detects your browser language and supports three languages:
- **English** (en)
- **中文** (zh-CN)
- **日本語** (ja)

To switch manually, use the language selector in the top-right corner. Your preference is saved in `localStorage`.

All schema labels, field descriptions, help texts, and choice labels are translated via the i18n overlay system.

### Model Settings

Configure model paths in the **Settings** dialog:

| Setting | Default | Description |
|---------|---------|-------------|
| **DiT model** | `models/diffusion_models/anima-base-v1.0.safetensors` | Base model path |
| **Qwen3 text encoder** | `models/text_encoders/qwen_3_06b_base.safetensors` | Text encoder path |
| **VAE** | `models/vae/qwen_image_vae.safetensors` | VAE path |

Paths resolve against the repository root (`ANIMA_HOME`). Set `ANIMA_DIT`, `ANIMA_VAE`, or `ANIMA_TEXT_ENCODER` environment variables to override.

### Hardware Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Mixed precision** | `bf16` | Training precision |
| **Attention mode** | `flash` | Attention implementation |

### Override Priority

Settings are applied in this order (later overrides earlier):

1. **Infrastructure defaults** — resolved from `library.env.resolve_under_home()`
2. **Infrastructure config** — per-workflow settings stored in `workflow.yaml`
3. **Stage config** — per-stage TOML overrides
4. **Auto-derived** — `network_weights`, `datasets`, etc. automatically filled from upstream outputs

Global settings (workflows root, etc.) are stored in `.anima_workflow_config.json` at the project root.
