# 双阶段 LoKR 训练工作流 — 端到端执行设计

## 目标

验证 Anima LoRA 管线的双阶段训练可行性：阶段1 训练 LoKR 到 epoch 6 → 使用其 checkpoint 初始化阶段2 → 阶段2 使用双数据集继续训练。

这不是平台验证脚本，而是正式的工作流执行任务。

## 数据集

`O:\LoRATraining\hanechan` — 8 个子目录（tree 模式子集），每个含 `.png` + `.txt`：
- `1_hanechan_v8_full` (14+3 images)
- `1_hanechan_v8_upper` (13+4 images)
- `3_hanechan_v8_lying` (6 images)
- `3_hanechan_v8_other` (7 images)
- `4_hanechan_v8_face` (6 images)
- `5_hanechan_v8_above` (5 images)
- `8_hanechan_v8_low` (3 images)
- `8_hanechan_v8_wide` (3 images)

## 四阶段工作流

### 阶段1：预处理 S1
- **脚本**: `scripts/preprocess/resize_images.py` + `cache_latents.py` + `cache_text_embeddings.py`
- **参数**: `--src O:/LoRATraining/hanechan --dst <run_dir>/preprocess_1/post_image_dataset --tree --bucket_families S1 --min_pixels 0`
- **产物**: `<run_dir>/preprocess_1/post_image_dataset/hanechan/<8个子目录>/.resized/` (PNG) + `.lora/` (_anima.npz, _anima_te.safetensors)
- **验证**: 每个子目录都有 `.resized/` 和 `.lora/`，且 `.lora/` 中有缓存文件

### 阶段2：训练1 (LoKR, stop@6)
- **脚本**: `train.py`
- **配置**: 通过 TOML 文件传递（`[[datasets.subsets]]` 不能通过命令行传递）
- **参数**:
  - `network_type = "lokr"`, `network_dim = 16`, `network_alpha = 8`, `lokr_factor = 8`
  - `decompose_both = true`, `use_tucker = true`, `scale_weight_norms = 1.0`
  - `max_train_epochs = 6`, `save_every_n_epochs = 6`（保证第6 epoch保存checkpoint）
  - `learning_rate = 0.0004`, `lr_scheduler = "cosine"`
  - `optimizer_type = "CAME"`, `optimizer_args = ["weight_decay=0.01", "betas=0.9,0.999,0.9999"]`
  - 数据集 = 预处理1 的 8 个子集（image_dir/cache_dir 指向预处理1的输出）
- **产物**: `<run_dir>/train_1/output/anima_lokr_epoch-000006.safetensors`
- **验证**: checkpoint 文件存在且非空

### 阶段3：预处理 S2
- 同阶段1，但 `--bucket_families S2`
- **独立目录**: `<run_dir>/preprocess_2/post_image_dataset/`
- **验证**: 同阶段1

### 阶段4：训练2 (LoKR, 从 Train1 checkpoint, 双数据集)
- **额外参数**:
  - `network_weights = "<train_1 checkpoint path>"`
  - `dim_from_weights = true`
  - 数据集 = 预处理1 (8子集) + 预处理2 (8子集) = 16 个 subsets
- **产物**: `<run_dir>/train_2/output/anima_lokr.safetensors`
- **验证**: 最终 checkpoint 存在且非空

## workflow 模块修正

### 1. PreprocessExecutor — 修正路径解析

**问题**: `_build_vae_cmd()` 和 `_build_te_cmd()` 缺少 `--qwen3` 和 `--vae` 的默认路径解析

**修正**: 从 `library/env.py` 的 `resolve_under_home()` 获取默认模型路径：
- VAE: `models/vae/` → `anima_lora/models/vae/`
- Qwen3: `models/qwen3/` → `anima_lora/models/qwen3/`

### 2. TrainExecutor — 改用 TOML 文件传递配置

**问题**: `_build_train_cmd()` 尝试将所有参数通过命令行传递，但 `[[datasets.subsets]]` 等嵌套结构无法通过 CLI 传递

**修正**: 
- `prepare_config()` 生成完整的 TOML 配置文件
- `_build_train_cmd()` 改为 `train.py --config <toml_file>` 的调用方式
- 检查 train.py 是否支持 `--config` 参数，如果不支持，创建 workflow 专用的包装脚本

### 3. 模型路径解析

**问题**: 基础设施配置中的模型路径可能为空

**修正**: 在 `workflow/stages/base.py` 中添加默认路径解析逻辑，使用 `library/env.py` 的 `resolve_under_home()`

## 实施步骤

1. 修正 `PreprocessExecutor` — 添加默认模型路径解析
2. 修正 `TrainExecutor` — 改用 TOML 文件传递配置
3. 检查/创建 `train.py` 的 `--config` 支持
4. 创建四阶段工作流配置（YAML + TOML）
5. 执行阶段1（预处理S1）→ 验证产物
6. 执行阶段2（训练1）→ 验证checkpoint
7. 执行阶段3（预处理S2）→ 验证产物
8. 执行阶段4（训练2）→ 验证最终产物
