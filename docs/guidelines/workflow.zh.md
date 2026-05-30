[English](workflow.md) | **中文**

# Workflow 引擎 — 自动化多阶段训练

Workflow 引擎是一个基于 aiohttp（后端）和 Vue 3 CDN（前端）构建的 WebUI + CLI 自动化训练流水线。它支持可配置的多阶段训练工作流，具备 Schema 驱动的动态表单、通过 SSE 实现实时进度反馈，以及跨阶段检查点续训功能。

## 安装

### Python 依赖

所有 Python 依赖均包含在 `pyproject.toml` 中。安装方式：

```bash
uv sync
```

关键依赖：
- `aiohttp >= 3.13.5` — HTTP 服务器和 REST API
- `pywebview >= 5.0` — 桌面窗口模式（可选，回退到浏览器）

### Node.js（仅开发时需要）

Workflow 前端通过 CDN 使用 Vue 3（生产环境无需构建步骤）。**只有在需要修改前端 JavaScript 时才需要安装 Node.js。**

从 [nodejs.org](https://nodejs.org/) 安装 Node.js（推荐 LTS 版本）或通过包管理器安装：

```bash
# Windows (winget)
winget install OpenJS.NodeJS.LTS

# macOS (Homebrew)
brew install node

# Linux (nvm - 推荐)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install --lts
```

前端开发时，可能需要带热重载的本地开发服务器：

```bash
cd workflow/web
npx serve .    # 或: python -m http.server 3000
```

### pywebview 系统依赖

在 Windows 上，pywebview 需要 **Microsoft Edge WebView2 Runtime**，该运行时已预装于 Windows 10 (1903+) 和 Windows 11。如果缺少，请从 [Microsoft](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) 下载。

在 Linux 上，pywebview 需要 `python3-gi` 或 `python3-pyqt5` — 参见 [pywebview 文档](https://pywebview.flowrl.com/guide/installation.html)。

### 启动

```bash
# 桌面窗口模式（默认）
python -m workflow

# 浏览器模式（无需 pywebview）
python -m workflow --no-gui

# 自定义端口和工作流根目录
python -m workflow --port 8765 --workflows-root /path/to/workflows
```

## 快速入门：单阶段训练

本示例演示如何创建一个基本的单阶段 LoKR 训练工作流。

### 1. 启动 Workflow UI

```bash
python -m workflow
```

桌面窗口将在 `http://localhost:8765` 打开。

### 2. 创建新工作流

点击 **"New Workflow"** 并为其命名（例如 `my_first_training`）。

### 3. 添加预处理阶段

1. 点击 **"Add Stage"** → 选择 **Preprocess**
2. 将 **Source directory** 设置为你的 `image_dataset/` 文件夹
3. 选择 **Bucket family** — 建议从 `L` 开始（1.03 MP，质量与速度的良好平衡）
4. **Min pixels** 保持默认值（500,000）

预处理阶段会将图像缩放到所选的 bucket family 尺寸，然后缓存 VAE 潜空间表示和文本嵌入。

### 4. 添加训练阶段

1. 点击 **"Add Stage"** → 选择 **Train**
2. 选择 **Method** — 例如 **LoKR**
3. 在 Schema 驱动的表单中配置参数（network_dim、learning_rate、max_train_epochs 等）
4. **Dataset** 字段会自动引用上游 Preprocess 阶段的输出

### 5. 运行

点击 **"Run"**。工作流将按顺序执行各阶段：

1. **Preprocess** — 缩放图像，缓存 VAE 潜空间表示和文本嵌入
2. **Train** — 训练 LoKR 适配器

### 6. 查找训练产物

训练输出按工作流目录组织：

```
.anima_workflow/my_first_training/
  runs/
    20260530-120000/          ← 带时间戳的运行目录
      preprocess_1/
        post_image_dataset/   ← 缩放后的图像和缓存
      train_1/
        output/
          *.safetensors       ← 你训练好的适配器
        command.txt           ← 实际执行的命令
        config.toml           ← 解析后的配置
      status.json             ← 运行状态快照
      run.log                 ← 完整日志
    latest → 20260530-120000/ ← 指向最新运行的 junction 链接
```

**三种方式查找最新的适配器：**

1. **`runs/latest/train_1/output/`** — `latest` junction 始终指向最近的运行
2. **History 标签页** — 点击任意已完成运行的 "Open directory" 按钮
3. **System log** — 训练完成时显示 safetensors 路径

## 单阶段使用详解

### 预处理阶段

| 设置 | 说明 |
|------|------|
| **Source directory** | 原始训练图像路径（需附带 `.txt` 标题侧栏文件） |
| **Bucket families** | 使用哪个分辨率族。详见 [Bucket Families 指南](bucket-families.zh.md)。 |
| **Min pixels** | 低于此像素数的图像将被跳过（默认：500,000） |

预处理阶段按顺序运行三个子步骤：
1. **Resize** — 缩放和裁剪图像以适配所选的 bucket family
2. **VAE cache** — 将图像编码到潜空间
3. **TE cache** — 将文本标题编码为嵌入向量

### 训练阶段

训练阶段呈现一个 Schema 驱动的表单，根据所选方法动态变化：

- **Method selector** — 在 LoRA、LoKR、LoHA 等之间切换的下拉菜单
- **Common parameters** — 学习率、epoch 数、批量大小、优化器
- **Method-specific parameters** — 例如 LoKR 的 `lokr_factor`，LoRA 的 `network_dim`

表单由 `workflow/schemas/train_{method}.yaml` 和 `workflow/schemas/train_common.yaml` 生成。

## 多阶段使用

多阶段工作流支持高级训练策略，例如[低分辨率预训练后接高分辨率精调](bucket-families.zh.md#multi-stage-training-strategy)。

### 多阶段编排的工作原理

阶段按 **拓扑顺序** 执行，顺序由 `depends_on` 声明决定。调度器会检测循环依赖并报告错误。

每个阶段的输出可通过以下方式供后续阶段使用：
- **自动引用** — 系统自动从上游输出填充 `network_weights` 和 `datasets`
- **占位符语法** — 配置值中的 `${stage_id.output_key}`，在运行时解析

### 多个预处理阶段

每个 Preprocess 阶段可以使用不同的设置：

| 设置 | Preprocess 1 | Preprocess 2 |
|------|-------------|-------------|
| **Bucket families** | `S1`（低分辨率，0.26 MP） | `L`（高分辨率，1.03 MP） |
| **Source directory** | `image_dataset/` | `image_dataset/`（相同或不同） |

这会生成两组不同分辨率的缓存数据，各自位于独立的子目录中。

### 多个训练阶段

#### `stop_epoch` — 中断并保存

在 Train 阶段设置 `stop_epoch` 可在指定 epoch 停止训练并确保保存检查点：

```
stop_epoch = 6
```

这会将 `max_train_epochs` 和 `save_every_n_epochs` 设置为指定值，使训练在保存 epoch-6 检查点后立即停止。

#### 检查点续训

当 Train 阶段在另一个 Train 阶段之后运行时，它会自动：

1. 找到上游阶段的 `safetensors_path` 输出
2. 将 `--network_weights` 设置为该路径
3. 对于 LoRA：设置 `--dim_from_weights` 以从检查点自动推断秩
4. 对于 LyCORIS（lokr/loha/locon）：设置 `dim_from_weights = false`（维度必须与配置匹配）

#### 典型的多阶段流程

```
Preprocess S1 → Train S1 (stop at epoch 6) → Preprocess L → Train L (from S1 checkpoint)
```

1. **Preprocess S1**：以 S1 family（0.26 MP）缩放并缓存
2. **Train S1**：训练 LoKR 适配器，在 epoch 6 停止
3. **Preprocess L**：以 L family（1.03 MP）缩放并缓存
4. **Train L**：从 S1 的 epoch-6 检查点继续，同时使用 S1 和 L 的缓存

第二个 Train 阶段通过占位符引用第一个的输出：`${train_1.safetensors_path}` → 解析为实际路径。

## 日志查看器

底部面板有三个标签页：

### System Log

显示工作流级别的事件：阶段开始/结束、检查点保存、错误。通过 SSE（Server-Sent Events）实时更新。

### Script Output

显示子进程的 stdout 输出：
- **TQDM 进度条** — 解析并以可视化进度条形式显示，包含步数、已用时间、预计剩余时间和指标（loss、lr）
- **阶段过滤** — 使用下拉菜单按阶段筛选输出
- **自动滚动** — 自动滚动到最新输出；通过滚动锁定按钮暂停/恢复
- **缓冲区限制** — 每阶段 500 行；超出时裁剪最早的行

### Run History

按时间倒序列出所有历史运行。每个条目显示：
- **Timestamp**（时间戳）和 **duration**（持续时间）
- **Status**（状态）：ok / stopped / error / running
- **Stage chain**（阶段链）带颜色编码的状态指示器
- **Actions**（操作）："View log" 和 "Open directory"

**从历史记录中查找最新的训练产物：**
1. 打开 **History** 标签页
2. 最近的运行在最上方
3. 点击 **"Open directory"** 打开运行文件夹
4. 导航到 `{train_stage_id}/output/` 查找 `.safetensors` 文件

或者，`runs/latest` 始终是指向最近运行目录的 junction/symlink。

### 搜索和高亮

日志查看器支持文本搜索，可在所有可见日志行中高亮匹配内容。

## 设置

### 语言

UI 会自动检测你的浏览器语言，支持三种语言：
- **English** (en)
- **中文** (zh-CN)
- **日本語** (ja)

要手动切换，请使用右上角的语言选择器。你的偏好会保存在 `localStorage` 中。

所有 Schema 标签、字段描述、帮助文本和选项标签均通过 i18n 覆盖系统进行翻译。

### 模型设置

在 **Settings** 对话框中配置模型路径：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| **DiT model** | `models/diffusion_models/anima-base-v1.0.safetensors` | 基础模型路径 |
| **Qwen3 text encoder** | `models/text_encoders/qwen_3_06b_base.safetensors` | 文本编码器路径 |
| **VAE** | `models/vae/qwen_image_vae.safetensors` | VAE 路径 |

路径相对于仓库根目录（`ANIMA_HOME`）解析。可设置 `ANIMA_DIT`、`ANIMA_VAE` 或 `ANIMA_TEXT_ENCODER` 环境变量进行覆盖。

### 硬件设置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| **Mixed precision** | `bf16` | 训练精度 |
| **Attention mode** | `flex` | Attention 实现方式 |

### 覆盖优先级

设置按以下顺序应用（后者覆盖前者）：

1. **Infrastructure defaults** — 从 `library.env.resolve_under_home()` 解析的基础设施默认值
2. **Infrastructure config** — 存储在 `workflow.yaml` 中的每工作流设置
3. **Stage config** — 每阶段的 TOML 覆盖
4. **Auto-derived** — `network_weights`、`datasets` 等自动从上游输出填充

全局设置（工作流根目录等）存储在项目根目录的 `.anima_workflow_config.json` 中。
