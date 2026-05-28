# Workflow 自动化训练引擎设计文档

## 1. 概述

### 1.1 目标

构建独立的 `workflow/` 模块，作为未来替代现有 PySide6 GUI 训练管理模式的自动化工作流引擎。提供 WebUI（Vue 3 CDN + aiohttp）和 CLI 两种交互方式，支持可配置的多阶段训练流水线。

### 1.2 核心能力

- 多阶段工作流编排（Preprocess → Train → Preprocess → Train ...）
- Schema 驱动的动态表单（独立参数定义，不依赖现有 configs/ 合并链）
- 实时进度反馈（SSE 流式推送）
- 跨阶段 checkpoint 权重传递（`--network_weights` + `--dim_from_weights`）
- 工作流版本管理和运行隔离

### 1.3 验证场景

使用 `O:\LoRATraining\hanechan` 数据集执行四阶段工作流：
1. Preprocess S1（bucket_families=S1）
2. Train S1（lokr, 10 epochs, stop at 6, cosine LR=0.0004）
3. Preprocess S2（bucket_families=S2）
4. Train S2（lokr, 使用 S1+S2 缓存, 4 epochs, constant LR≈0.000138, 衔接 Train S1 的 epoch 6 checkpoint）

---

## 2. Shared Tensor 可行性分析

### 2.1 两类共享机制

| 类型 | 机制 | 保存/加载行为 |
|------|------|-------------|
| Non-persistent buffer（sigma/fei/routing） | `_wire_shared_*` 通过 `module._buffers[name] = shared_tensor` 建立引用别名 | `persistent=False` → 不进入 `state_dict()` → `load_state_dict()` 不触碰 → 别名完好 |
| Persistent 参数（lambda_layer/S_p/S_q） | PyTorch 内部优化导致 storage 共享 | `save_weights()` 路径经 `detach().clone()` 独立保存 → 阶段2各模块独立拥有副本 |

### 2.2 结论

Shared tensor 不会导致双阶段方案失败：
1. 需要保持共享的 buffer 是 `persistent=False`，阶段2 `__init__` 中 `_wire_shared_*` 重建别名
2. `set_sigma` 等方法内置 aliasing-recovery（identity check + rebind），即使 `.to(device)` 破坏别名也能自动恢复
3. Persistent 参数在阶段2中各自独立是正确行为——梯度更新互不影响
4. "Removed shared tensor" 警告仅出现在 `accelerator.save_state()` 路径，不出现在 `save_every_n_epochs` 路径

---

## 3. 架构设计

### 3.1 技术栈

| 层 | 选择 | 理由 |
|---|---|---|
| HTTP 服务 | aiohttp（项目已有依赖） | 无需新增依赖 |
| 前端框架 | Vue 3 CDN 模式 | 无需 node.js 构建工具链，组件化+响应式 |
| 桌面窗口 | pywebview（需新增） + `--no-gui` 回退 | 双模式：桌面窗口或纯浏览器 |
| 参数定义 | YAML Schema 文件 | 独立于现有 configs/，前后端单一事实来源 |
| 配置存储 | YAML（工作流定义）+ TOML（阶段参数） | 与项目风格一致 |

### 3.2 模块目录结构

```
workflow/
├── __init__.py
├── __main__.py              # python -m workflow 入口（双模式）
├── app.py                   # aiohttp Application + 路由注册
├── scheduler.py             # 工作流调度器（核心引擎）
├── models.py                # Pydantic 数据模型
├── config.py                # YAML/TOML 配置读写 + 占位符替换
├── logger.py                # 统一日志记录器
│
├── schemas/                 # 参数 Schema 定义
│   ├── preprocess.yaml
│   ├── train_lokr.yaml
│   └── train_lora.yaml
│
├── stages/
│   ├── __init__.py
│   ├── base.py              # 阶段基类（prepare/execute/get_outputs）
│   ├── preprocess.py        # 预处理执行器
│   └── train.py             # 训练执行器
│
├── web/
│   ├── index.html           # SPA 入口
│   ├── css/style.css
│   └── js/
│       ├── app.js           # Vue 3 主入口
│       ├── api.js           # HTTP API 封装
│       └── components/
│           ├── StageList.js
│           ├── StageCard.js
│           ├── SchemaForm.js
│           ├── FieldRenderer.js
│           ├── DatasetSelector.js
│           ├── RunControl.js
│           └── LogViewer.js
│
├── scripts/
│   ├── run_workflow.py      # CLI: 运行工作流
│   └── create_workflow.py   # CLI: 创建新工作流
│
└── templates/               # 默认配置模板
    ├── preprocess_default.toml
    └── train_lokr_default.toml
```

### 3.3 数据流

```
WebUI (Vue 3 CDN)
  │
  │ REST API + SSE
  ▼
aiohttp Server (app.py)
  │
  ▼
Scheduler (scheduler.py)
  │ 读取 workflow.yaml → 拓扑排序
  │ 替换占位符 → 写入 config.toml
  │ 调用 StageExecutor
  ▼
Stage Executors (stages/)
  ├─ PreprocessExecutor: resize → VAE cache → TE cache
  └─ TrainExecutor: 组装 CLI 参数 → subprocess(train.py)
```

---

## 4. 参数 Schema 设计

### 4.1 设计原则

- **独立于 configs/**：不依赖现有三层合并链，Workflow 有自己的参数定义
- **三种分类**：必选（required）、条件必选（conditional_required）、可选（optional）
- **前后端共享**：Schema YAML 是单一事实来源，前端生成表单，后端验证参数
- **模板导入**：支持从现有 configs/gui-methods/*.toml 导入作为蓝本
- **参数分层**：通用参数与方法特定参数分离，切换方法时通用参数保留
- **基础设施分离**：模型路径等全局配置不属于阶段配置，属于工作流级基础设施设置

### 4.2 参数分层架构

Train 阶段的参数分为三个层次，每层独立管理：

```
┌─────────────────────────────────────────────────────────┐
│  工作流级基础设施配置（infrastructure.yaml）               │
│  所有阶段共享，不属于任何单个阶段                           │
│  ├─ 模型路径: pretrained_model_name_or_path, qwen3, vae  │
│  └─ 硬件默认: mixed_precision, attn_mode                  │
├─────────────────────────────────────────────────────────┤
│  Train 通用参数层（train_common.yaml）                    │
│  所有训练方法共享，切换方法时保留                           │
│  ├─ 训练超参: learning_rate, epochs, optimizer, scheduler │
│  ├─ 性能: gradient_checkpointing, blocks_to_swap          │
│  ├─ 数据: caption_dropout_rate, cache_llm_adapter_outputs │
│  ├─ 输出: output_name, save_every_n_epochs                │
│  └─ 衔接: network_weights, dim_from_weights               │
├─────────────────────────────────────────────────────────┤
│  方法特定参数层（train_lokr.yaml, train_lora.yaml, ...）  │
│  仅在选择对应方法时加载，切换方法时替换                     │
│  ├─ 网络类型: network_type (lora/lokr/loha/locon)         │
│  ├─ 维度: network_dim, network_alpha, conv_dim, conv_alpha│
│  ├─ LyCORIS: lokr_factor, decompose_both, use_tucker, ...│
│  ├─ Ortho: use_ortho, ortho_init_std                      │
│  ├─ MoE: use_moe_style, num_experts, router_source, ...  │
│  └─ 附加: scale_weight_norms, use_timestep_mask, ...      │
└─────────────────────────────────────────────────────────┘
```

### 4.3 方法选择机制

**网络类型的确定逻辑**（与 train.py 的 `resolve_network_spec()` 分派优先级一致）：

1. **基础类型**：`network_type` 直接选择（`loha` / `locon` / `lokr`），不与 MoE/ortho 组合
2. **LoRA 变体**：当 `network_type` 为空或 `lora` 时，通过组合标志派生：
   - `use_chimera_hydra=true` → ChimeraHydra
   - `use_moe_style="independent_A"` → FeRA/StackedExperts
   - `use_moe_style="shared_A"` + `use_ortho=true` → OrthoHydra
   - `use_moe_style="shared_A"` → HydraLoRA
   - `use_ortho=true` → OrthoLoRA
   - 默认 → LoRA

**前端交互**：用户先选择基础类型（LoRA / LoHA / LoCON / LoKR），再通过组合开关（Ortho / MoE / T-LoRA / ReFT）叠加变体。Schema 中标记 `combo: true` 的字段可跨方法组合。

**切换方法时的参数保留**：前端维护一个 `commonParams` 对象存储通用参数值。切换方法时，仅替换方法特定参数层，通用参数从 `commonParams` 恢复。

### 4.4 Schema 字段定义

```yaml
fields:
  - key: string              # 参数键名（对应 train.py 的 argparse 参数）
    type: string              # int | float | str | bool | enum | path | list[str] | dataset_ref | checkpoint_ref
    layer: string             # "common" | "method" | "infrastructure" — 参数所属层
    required: boolean         # 是否必选
    conditional_required: boolean  # 是否条件必选
    condition: string         # 条件表达式
    combo: boolean            # 是否可跨方法组合（如 use_ortho, add_reft）
    auto_set: string | null   # 在特定条件下自动设置值（如 loha/lokr 自动开启 scale_weight_norms）
    default: any              # 默认值
    choices: list             # enum 类型的可选值
    label: string             # 前端显示标签
    help: string              # 帮助文本
    hidden: boolean           # 是否在表单中隐藏
    widget: string            # 特殊控件类型
    group: string             # 所属分组
```

### 4.5 参数分类的视觉区分

| 分类 | 标记 | 行为 |
|------|------|------|
| 必选 | 红色 `*` | 空值时红色边框 + 提示，阻止运行 |
| 条件必选 | 橙色 `*` | 条件满足时显示并标为必选 |
| 自动设置 | 蓝色 `⚡` | 由系统自动设置值，用户可覆盖（如 loha/lokr 的 scale_weight_norms） |
| 可选 | 无标记 | 有默认值，不填不影响运行 |

### 4.6 字段控件映射

| Schema 类型 | 前端控件 |
|------------|---------|
| `int` | `<input type="number">` |
| `float` | `<input type="text">` + 数字校验 |
| `str` | `<input type="text">` |
| `bool` | 开关按钮 |
| `enum` | `<select>` 下拉框 |
| `path` | `<input>` + 浏览按钮 |
| `list[str]` | 标签输入框 |
| `dataset_ref` | 复选框列表（列出可用 Preprocess 阶段输出） |
| `checkpoint_ref` | 单选按钮列表（列出可用 Train 阶段 safetensors） |
| `method_selector` | 基础类型选择 + 组合开关面板 |

### 4.7 工作流级基础设施配置

文件 `workflow/schemas/infrastructure.yaml`，所有阶段共享：

```yaml
type: infrastructure
label: "基础设施配置"
description: "所有阶段共享的模型路径和硬件设置"

groups:
  - name: models
    label: "模型路径"
    fields:
      - key: pretrained_model_name_or_path
        type: path
        required: false
        layer: infrastructure
        label: "DiT 模型路径"
        help: "留空使用默认路径 (models/diffusion_models/)"
      - key: qwen3
        type: path
        required: false
        layer: infrastructure
        label: "文本编码器路径"
        help: "留空使用默认路径"
      - key: vae
        type: path
        required: false
        layer: infrastructure
        label: "VAE 模型路径"
        help: "留空使用默认路径"

  - name: hardware
    label: "硬件设置"
    fields:
      - key: mixed_precision
        type: enum
        required: false
        default: "bf16"
        choices: ["no", "fp16", "bf16"]
        layer: infrastructure
        label: "混合精度"
      - key: attn_mode
        type: enum
        required: false
        default: "flex"
        choices: ["flex", "flash"]
        layer: infrastructure
        label: "注意力模式"
```

前端在工作流设置页面（非阶段配置）中提供基础设施配置入口。这些参数在调度器运行时自动注入所有阶段的配置。

### 4.8 Train 通用参数层

文件 `workflow/schemas/train_common.yaml`，所有训练方法共享：

```yaml
type: train_common
label: "训练通用参数"
description: "所有训练方法共享的参数，切换方法时保留"

groups:
  - name: training
    label: "训练超参"
    fields:
      - key: learning_rate
        type: float
        required: true
        layer: common
        default: 0.0004
        label: "学习率"
      - key: lr_scheduler
        type: enum
        required: false
        layer: common
        default: "cosine"
        choices: ["cosine", "constant", "constant_with_warmup", "linear", "polynomial"]
        label: "LR 调度器"
      - key: lr_warmup_steps
        type: int
        required: false
        layer: common
        default: 0
        label: "预热步数"
      - key: max_train_epochs
        type: int
        required: true
        layer: common
        label: "最大 Epoch 数"
      - key: stop_epoch
        type: int
        required: false
        layer: common
        conditional_required: true
        condition: "max_train_epochs > 0"
        label: "停止 Epoch"
        help: "在此 epoch 保存后停止"
      - key: optimizer_type
        type: enum
        required: false
        layer: common
        default: "CAME"
        choices:
          - "AdamW"
          - "AdamW8bit"
          - "Lion"
          - "CAME"
          - "Prodigy"
          - "Prodigy_Adv"
          - "Adopt_Adv"
          - "Adafactor"
          - "RAdamScheduleFree"
          - "AdamWScheduleFree"
          - "SGDScheduleFree"
          - "DAdaptAdam"
          - "DAdaptSGD"
          - "PagedAdamW"
          - "PagedAdamW8bit"
        label: "优化器"
      - key: optimizer_args
        type: str
        required: false
        layer: common
        label: "优化器参数"
        help: "如 weight_decay=0.01, betas=0.9,0.999,0.9999"
      - key: gradient_accumulation_steps
        type: int
        required: false
        layer: common
        default: 1
        label: "梯度累积步数"
      - key: max_grad_norm
        type: float
        required: false
        layer: common
        default: 1.0
        label: "最大梯度范数"

  - name: data
    label: "数据"
    fields:
      - key: datasets
        type: dataset_ref
        required: true
        layer: common
        label: "输入数据集"
      - key: use_shuffled_caption_variants
        type: bool
        required: false
        layer: common
        default: true
        label: "使用打乱的 Caption 变体"
      - key: caption_dropout_rate
        type: float
        required: false
        layer: common
        default: 0.03
        label: "Caption Dropout 率"
      - key: cache_llm_adapter_outputs
        type: bool
        required: false
        layer: common
        default: true
        label: "缓存 LLM Adapter 输出"

  - name: checkpoint
    label: "Checkpoint 衔接"
    fields:
      - key: network_weights
        type: checkpoint_ref
        required: false
        layer: common
        conditional_required: true
        condition: "has_upstream_train == true"
        label: "权重初始化"
      - key: dim_from_weights
        type: bool
        required: false
        layer: common
        default: true
        hidden: true

  - name: output
    label: "输出"
    fields:
      - key: output_name
        type: str
        required: false
        layer: common
        default: "anima_lora"
        label: "输出名称"
      - key: save_every_n_epochs
        type: int
        required: false
        layer: common
        label: "每N个Epoch保存"
      - key: checkpointing_epochs
        type: int
        required: false
        layer: common
        label: "可恢复检查点间隔"

  - name: performance
    label: "性能"
    collapsed: true
    fields:
      - key: gradient_checkpointing
        type: bool
        required: false
        layer: common
        default: false
        label: "梯度检查点"
      - key: unsloth_offload_checkpointing
        type: bool
        required: false
        layer: common
        default: false
        label: "Unsloth 卸载检查点"
      - key: blocks_to_swap
        type: int
        required: false
        layer: common
        default: 0
        label: "Block Swap 数量"
      - key: torch_compile
        type: bool
        required: false
        layer: common
        default: false
        label: "torch.compile 加速"
      - key: seed
        type: int
        required: false
        layer: common
        label: "随机种子"
```

### 4.9 方法特定参数层

每个方法有一个独立的 Schema 文件，仅在选择该方法时加载。

#### LoKR 方法特定参数（`train_lokr.yaml`）

```yaml
type: train_method
method: lokr
label: "LoKR"
description: "Low-Rank Kronecker Product"
base_type: lycoris

groups:
  - name: architecture
    label: "架构"
    fields:
      - key: network_type
        type: str
        layer: method
        required: true
        default: "lokr"
        hidden: true
      - key: network_dim
        type: int
        layer: method
        required: true
        default: 16
        label: "Network Dim"
      - key: network_alpha
        type: int
        layer: method
        required: true
        default: 8
        label: "Network Alpha"
      - key: conv_dim
        type: int
        layer: method
        required: false
        default: 1
        label: "Conv Dim"
      - key: conv_alpha
        type: int
        layer: method
        required: false
        default: 4
        label: "Conv Alpha"
      - key: lokr_factor
        type: int
        layer: method
        required: false
        default: 8
        label: "LoKr Factor"
        help: "Kronecker 分解因子，-1 为自动"
      - key: decompose_both
        type: bool
        layer: method
        required: false
        default: true
        label: "Decompose Both"
      - key: use_tucker
        type: bool
        layer: method
        required: false
        default: true
        label: "Use Tucker"
      - key: use_scalar
        type: bool
        layer: method
        required: false
        default: false
        label: "Use Scalar"
      - key: weight_decompose
        type: bool
        layer: method
        required: false
        default: false
        label: "Weight Decompose (DoRA)"
      - key: full_matrix
        type: bool
        layer: method
        required: false
        default: false
        label: "Full Matrix"
      - key: scale_weight_norms
        type: float
        layer: method
        required: false
        default: 1.0
        auto_set: "network_type in ('loha', 'lokr') → 1.0"
        label: "Scale Weight Norms"
        help: "权重范数正则化。LoHA/LoKR 自动启用为 1.0"
```

#### LoRA 方法特定参数（`train_lora.yaml`）

```yaml
type: train_method
method: lora
label: "LoRA"
description: "经典 LoRA，支持 Ortho/Hydra/ReFT/T-LoRA 组合"
base_type: lora
supports_combos: true

combo_switches:
  - key: use_ortho
    label: "OrthoLoRA"
    description: "Cayley/SVD 正交参数化"
    schema_file: "combo_ortho.yaml"
  - key: use_moe_style
    label: "HydraLoRA MoE"
    description: "多专家路由"
    schema_file: "combo_hydra.yaml"
    value: "shared_A"
  - key: use_timestep_mask
    label: "T-LoRA"
    description: "时间步掩码"
    schema_file: "combo_tlora.yaml"
  - key: add_reft
    label: "ReFT"
    description: "表示微调"
    schema_file: "combo_reft.yaml"

groups:
  - name: architecture
    label: "架构"
    fields:
      - key: network_type
        type: str
        layer: method
        required: false
        hidden: true
      - key: network_dim
        type: int
        layer: method
        required: true
        default: 32
        label: "Network Dim"
      - key: network_alpha
        type: int
        layer: method
        required: true
        default: 32
        label: "Network Alpha"
      - key: network_dropout
        type: float
        layer: method
        required: false
        label: "Dropout"
      - key: scale_weight_norms
        type: float
        layer: method
        required: false
        label: "Scale Weight Norms"
        help: "权重范数正则化，留空不启用"
```

#### LoHA / LoCON 方法（`train_loha.yaml`, `train_locon.yaml`）

结构类似 LoKR，但方法特定字段不同。LoHA 有 `use_tucker` 但无 `lokr_factor`/`decompose_both`。LoCON 有 `use_tucker` 和 `conv_dim`/`conv_alpha`。

### 4.10 前端表单渲染逻辑

```
用户创建 Train 阶段
    │
    ├─ 1. 显示方法选择器
    │   ├─ 基础类型: LoRA / LoHA / LoCON / LoKR
    │   └─ 组合开关: Ortho / MoE / T-LoRA / ReFT（仅 LoRA 基础类型可用）
    │
    ├─ 2. 加载参数层
    │   ├─ 始终加载: train_common.yaml → 通用参数表单
    │   └─ 按选择加载: train_{method}.yaml → 方法特定参数表单
    │
    ├─ 3. 组合开关交互
    │   ├─ 开启 Ortho → 追加 combo_ortho.yaml 的字段
    │   ├─ 开启 MoE → 追加 combo_hydra.yaml 的字段
    │   └─ 关闭某开关 → 移除对应字段，但通用参数保留
    │
    ├─ 4. 切换方法时
    │   ├─ 保存当前 commonParams 到临时对象
    │   ├─ 清空方法特定字段
    │   ├─ 加载新方法的 Schema
    │   └─ 从临时对象恢复 commonParams
    │
    └─ 5. 自动设置处理
        ├─ 选择 LoKR → scale_weight_norms 自动设为 1.0，标记为 ⚡
        └─ 用户手动修改 → 覆盖自动值，标记变为手动
```

### 4.11 Preprocess Schema

```yaml
type: preprocess
label: "预处理"
description: "图像缩放 + VAE缓存 + 文本嵌入缓存"

groups:
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
        label: "Bucket Families"
        choices: ["S1", "S2", "XS", "S", "M", "L", "XL"]
        default: ["S1"]

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

---

## 5. UI/UX 设计

### 5.1 页面布局

三区域布局：左侧阶段面板 + 右侧配置面板 + 底部日志/运行控制

```
┌──────────────────────────────────────────────────────────────────┐
│  🔄 Anima Workflow                           [打开工作流 ▾]      │
│                                               [新建工作流]       │
├──────────────────────────┬───────────────────────────────────────┤
│  阶段面板（可拖拽排序）   │  配置面板（Schema 驱动动态表单）       │
│                           │                                       │
│  ┌────────────────────┐   │  当前选中阶段表单                     │
│  │ 📁 Preprocess S1 ✅│   │  Basic / Advanced 折叠分组             │
│  ├────────────────────┤   │  必选/条件必选/可选 视觉区分           │
│  │ 🎯 Train S1     🔄│   │  dataset_ref / checkpoint_ref 选择器  │
│  ├────────────────────┤   │                                       │
│  │ 📁 Preprocess S2 ⏳│   │  [保存配置] [从模板导入 ▾]            │
│  ├────────────────────┤   │                                       │
│  │ 🎯 Train S2     ⏳│   │                                       │
│  └────────────────────┘   │                                       │
│  [+ 添加阶段] ▾           │                                       │
│  ─── 运行控制 ─────       │                                       │
│  [▶ 运行] [■ 停止]       │                                       │
│  ▓▓▓▓▓░░░░ 60% 总进度    │                                       │
│  ─── 日志 ────────       │                                       │
│  [时间戳] 日志行...       │                                       │
├──────────────────────────┴───────────────────────────────────────┤
│  （底部日志区域可拖拽调整高度）                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 阶段交互

1. **添加阶段**：点击 `[+ 添加阶段]` → 下拉选择 Preprocess/Train → 面板底部新增卡片 → 自动选中 → 右侧加载 Schema 表单
2. **阶段卡片**：显示类型图标 + 自动编号 + 状态 + 迷你进度条（运行中时）
3. **拖拽排序**：上下拖动调整阶段顺序
4. **右键菜单**：复制配置 / 删除 / 重命名
5. **选中**：点击卡片 → 右侧显示配置表单

### 5.3 配置复用

点击 `[从模板导入 ▾]` 提供：
- **内置模板**：`workflow/templates/` 下的默认配置
- **已有阶段复制**：当前工作流中同类型阶段的配置
- **外部 TOML 导入**：读取 `configs/gui-methods/*.toml`，自动匹配 Schema key，显示匹配/未匹配结果预览

### 5.4 数据集选择器交互

Train 阶段的数据集配置区域：

```
┌─ 数据集 ──────────────────────────────────────────┐
│  输入数据集:                                        │
│  ☑ 📁 Preprocess S1 (S1, 7 subsets)               │
│  ☑ 📁 Preprocess S2 (S2, 7 subsets)               │
│                                                     │
│  子集详情（展开）:                                   │
│  Preprocess S1:                                     │
│    ├─ 1_hanechan_v8_full    repeats=1              │
│    ├─ 3_hanechan_v8_lying   repeats=3              │
│    └─ ...                                           │
│  Preprocess S2:                                     │
│    ├─ 1_hanechan_v8_full    repeats=1              │
│    └─ ...                                           │
│                                                     │
│  num_repeats 设置:                                   │
│  ○ 自动（从源目录前缀数字推断）                      │
│  ● 自定义 [编辑表格]                                │
└─────────────────────────────────────────────────────┘
```

子集列表来自预处理阶段的实际产物（调度器扫描 `.lora/` 目录得到），不是预设。

---

## 6. 预处理目录结构与多数据集合并

### 6.1 实际产物目录结构

每个 Preprocess 阶段在运行目录下产出一棵完整的树形目录：

```
runs/20260528_100000/
├── preprocess_s1/
│   ├── config.toml
│   ├── stage_outputs.yaml           # 产物清单
│   └── post_image_dataset/
│       └── hanechan/
│           ├── 1_hanechan_v8_full/
│           │   ├── .resized/        # image_dir（缩放后 PNG）
│           │   └── .lora/           # cache_dir（NPZ + TE）
│           ├── 3_hanechan_v8_lying/
│           │   ├── .resized/
│           │   └── .lora/
│           └── ... (auto_scan 产生的子集)
│
├── preprocess_s2/                   # 完全独立，不覆盖 S1
│   ├── config.toml
│   ├── stage_outputs.yaml
│   └── post_image_dataset/
│       └── hanechan/
│           ├── 1_hanechan_v8_full/
│           │   ├── .resized/
│           │   └── .lora/
│           └── ...
```

### 6.2 关键约束

- `.resized/` 中的 PNG 文件名不含 bucket 信息，不同 bucket_families 的预处理**会互相覆盖**
- 因此 S1 和 S2 **必须使用独立目录树**，不能共享同一输出路径
- NPZ 文件名包含分辨率（如 `img_0512x0512_anima.npz`），不同 bucket 产生不同分辨率的 NPZ 文件可共存
- TE 缓存文件名不含分辨率（`img_anima_te.safetensors`），但 TE 编码不依赖 bucket 分辨率，覆盖无功能影响

### 6.3 S1+S2 合并方式

Train S2 的 TOML 配置中，`[[datasets.subsets]]` 同时引用两棵独立目录树的子集：

```toml
# S1 子集
[[datasets]]
validation_split_num = 0

[[datasets.subsets]]
image_dir = "${preprocess_s1.dataset_dir}/hanechan/1_hanechan_v8_full/.resized"
cache_dir = "${preprocess_s1.dataset_dir}/hanechan/1_hanechan_v8_full/.lora"
num_repeats = 1
recursive = true

# S2 子集
[[datasets]]
validation_split_num = 0

[[datasets.subsets]]
image_dir = "${preprocess_s2.dataset_dir}/hanechan/1_hanechan_v8_full/.resized"
cache_dir = "${preprocess_s2.dataset_dir}/hanechan/1_hanechan_v8_full/.lora"
num_repeats = 1
recursive = true
```

### 6.4 子集发现

Preprocess 阶段完成后，调度器扫描产物目录发现子集：

```python
def discover_preprocess_outputs(stage_dir: Path) -> list[SubsetInfo]:
    post_dir = stage_dir / "post_image_dataset"
    subsets = []
    for lora_dir in sorted(post_dir.rglob(".lora")):
        subset_name = lora_dir.parent.name
        resized_dir = lora_dir.parent / ".resized"
        if resized_dir.exists():
            num_repeats = parse_repeats_from_name(subset_name)
            subsets.append(SubsetInfo(
                name=subset_name,
                image_dir=str(resized_dir),
                cache_dir=str(lora_dir),
                num_repeats=num_repeats,
            ))
    return subsets
```

结果存入 `stage_outputs.yaml` 供后续 Train 阶段引用。

---

## 7. 进度报告

### 7.1 SSE 事件格式

```
GET /api/runs/{run_id}/events (text/event-stream)
```

事件类型：

| 事件 | 字段 | 说明 |
|------|------|------|
| `workflow_start` | `total_stages` | 工作流开始 |
| `stage_start` | `stage_id`, `stage_type` | 阶段开始 |
| `stage_progress` | `stage_id`, `pct`, `cur`, `total`, `desc`, `rate`, `eta`, `epoch?`, `loss?`, `lr?` | 进度更新 |
| `stage_ckpt` | `stage_id`, `path`, `epoch` | 保存 checkpoint |
| `stage_end` | `stage_id`, `status` | 阶段结束 |
| `workflow_end` | `status` | 工作流结束 |

### 7.2 进度捕获机制

**Preprocess 阶段**：
- subprocess stdout → tqdm 行正则解析 → `stage_progress` 事件
- 三步子进度：Resizing → Caching latents → Caching text embeddings

**Train 阶段**：
- 主通道：增量读取 `progress.jsonl`（复用 train.py 的 ProgressSink 格式，无需修改 train.py）
- 回退通道：stdout tqdm 行解析

### 7.3 前端进度 UI

每个阶段卡片显示状态图标 + 进度条 + 速率 + ETA：

```
🔄 Train S1   ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░ 60%
   step 600/1000 · epoch 6/10 · loss 0.123 · 1.0s/step · ETA 06:40
```

状态：⏳等待 / ▶启动（跑马灯） / 🔄运行（进度条） / ✅完成 / ❌失败

---

## 8. 调度器工作循环

```
用户点击 [▶ 运行]
    │
    ├─ 1. 前端验证（必选参数、依赖关系）
    ├─ 2. POST /api/workflows/{name}/run
    ├─ 3. 后端创建 runs/{timestamp}/ 目录
    ├─ 4. 按拓扑排序执行每个阶段：
    │   a. 替换占位符 → 写入 config.toml
    │   b. 创建 StageExecutor
    │   c. 启动 subprocess
    │   d. StageProgressWatcher 捕获进度 → SSE 推送
    │   e. 等待完成 → 验证输出产物
    │   f. 失败则中止工作流
    └─ 5. workflow_end 事件 → 前端更新最终状态
```

### 特殊处理

| 场景 | 调度器行为 |
|------|----------|
| `stop_epoch` 设置 | 将 `max_train_epochs` 覆盖为 `stop_epoch` 值，设置 `save_every_n_epochs = stop_epoch` |
| `network_weights` 设置 | 添加 `--network_weights` + `--dim_from_weights` 参数 |
| Preprocess 子集引用 | 从 `stage_outputs.yaml` 读取子集列表，生成 `[[datasets.subsets]]` |
| `--method` / `--preset` | Train 阶段的 TOML 配置中包含 `method` 和 `preset` 字段，调度器将其转换为 train.py 的 CLI 参数。由于 Workflow 使用独立参数定义（不走三层合并链），所有参数直接写入阶段 TOML，train.py 通过 `--config_file` 加载 |

### 预处理阶段调用方式

预处理执行器通过 subprocess 直接调用底层脚本（不经过 `tasks.py`）：

```
resize:       python scripts/preprocess/resize_images.py --src <source> --dst <output> --tree --bucket_families S1
vae_cache:    python scripts/preprocess/cache_latents.py --dir <output> --cache_dir <output> --tree
te_cache:     python scripts/preprocess/cache_text_embeddings.py --dir <source_subset> --cache_dir <cache_subset> --recursive
```

### 训练阶段调用方式

训练执行器通过 subprocess 直接调用 train.py：

```
python train.py --config_file <resolved_config.toml> --dataset_config <resolved_config.toml> --output_dir <run_dir/train_s1/output>
```

阶段 TOML 包含所有参数（不走 base→preset→method 合并链），`--config_file` 让 train.py 直接加载完整配置。

---

## 9. 工作流目录结构

```
workflows/
└── hanechan-lokr-two-stage/
    ├── workflow.yaml                 # 当前工作流设计
    ├── configs/                      # 阶段配置（含占位符）
    │   ├── preprocess_s1.toml
    │   ├── train_s1.toml
    │   ├── preprocess_s2.toml
    │   └── train_s2.toml
    ├── history/                      # 历史版本
    │   └── workflow_20260528_100000.yaml
    └── runs/
        └── 20260528_100000/
            ├── workflow_resolved.yaml
            ├── run.log
            ├── preprocess_s1/
            │   ├── config.toml
            │   ├── stage_outputs.yaml
            │   └── post_image_dataset/hanechan/...
            ├── train_s1/
            │   ├── config.toml
            │   ├── output/*.safetensors
            │   └── train.log
            ├── preprocess_s2/
            │   └── ...
            └── train_s2/
                ├── config.toml
                ├── output/*.safetensors
                └── train.log
```

---

## 10. REST API 设计

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/workflows` | 列出工作流 |
| POST | `/api/workflows` | 创建工作流 |
| GET | `/api/workflows/{name}` | 获取工作流详情 |
| PUT | `/api/workflows/{name}` | 更新工作流 |
| DELETE | `/api/workflows/{name}` | 删除工作流 |
| POST | `/api/workflows/{name}/run` | 运行工作流 |
| POST | `/api/workflows/{name}/stop` | 停止运行 |
| GET | `/api/runs/{run_id}/events` | SSE 流式事件 |
| GET | `/api/runs/{run_id}/log` | 获取日志 |
| GET | `/api/schemas/{stage_type}` | 获取参数 Schema |
| GET | `/api/templates` | 列出配置模板 |
| POST | `/api/import-toml` | 从 TOML 导入配置 |
| GET | `/` | WebUI 静态页面 |

---

## 11. UI 视觉标准

### 11.1 暗色主题配色（与现有 GUI 一致）

| 角色 | 颜色 | 用途 |
|------|------|------|
| 窗口背景 | `#1e1e1e` | 主背景 |
| 面板背景 | `#232323` | 侧边栏、卡片背景 |
| 输入框背景 | `#2a2a2a` | 文本框、下拉框、日志区域 |
| 文本颜色 | `#dcdcdc` | 主要文本 |
| 高亮色 | `#3c78c8` | 选中项、活动标签页 |
| 边框色 | `#555555` | 输入框、卡片、分割线 |

### 11.2 按钮颜色语义

| 语义 | 颜色 | 用于 |
|------|------|------|
| 正面操作 | `#27ae60` 绿色 | 运行、保存 |
| 负面操作 | `#c0392b` 红色 | 停止 |
| 次要操作 | `#2980b9` 蓝色 | 添加阶段、导入 |
| 未保存状态 | `#e67e22` 橙色 | 保存 * |
| 禁用状态 | `#7f8c8d` 灰色 | 运行中的不可用按钮 |

### 11.3 字体

| 区域 | 字体 | 大小 |
|------|------|------|
| 全局 UI | system-ui, sans-serif | 13px |
| 日志/代码 | monospace | 12px |
| 标题 | system-ui, sans-serif | 16px, font-weight: 600 |
| 标签页 | system-ui, sans-serif | 13px, font-weight: 500 |
| 小字/辅助 | system-ui, sans-serif | 11px |

### 11.4 间距

- 卡片间距：8px
- 面板内边距：12px
- 表单行间距：6px
- 分组标题上边距：16px

---

## 12. 工作流打开与保存

### 12.1 打开工作流

**操作方式**：打开工作流**目录**（不是单个文件）。

```
用户点击 [打开工作流 ▾]
    │
    ├─ 选项 1: 从最近列表选择
    │   └─ 下拉显示最近打开的 5 个工作流路径
    │
    ├─ 选项 2: 浏览目录
    │   └─ 弹出系统目录选择对话框
    │   └─ 选择包含 workflow.yaml 的目录
    │
    └─ 选项 3: 新建工作流
        └─ 弹出对话框：输入名称 + 选择存储位置
        └─ 创建目录结构 + 空 workflow.yaml
```

**验证逻辑**：
1. 选中目录后检查 `workflow.yaml` 是否存在
2. 不存在 → 提示"此目录不是有效的工作流目录，是否创建新工作流？"
3. 存在 → 加载 workflow.yaml + configs/*.toml → 恢复阶段面板状态

### 12.2 保存工作流

- **自动保存**：每次修改阶段配置时自动保存到 workflow.yaml + configs/*.toml
- **手动保存**：`[保存]` 按钮（有未保存更改时显示橙色 `保存 *`）
- **另存为**：`[另存为]` → 复制整个工作流目录到新位置
- **历史版本**：每次保存时自动将当前 workflow.yaml 备份到 `history/` 目录

### 12.3 启动时恢复

Workflow 应用启动时：
1. 检查是否有上次打开的工作流（存储在 localStorage）
2. 有 → 自动加载
3. 无 → 显示欢迎页面（创建新工作流 / 打开已有工作流）

---

## 13. Loss 曲线可视化

### 13.1 数据来源与特征

**数据来自 train.py 的 `progress.jsonl`**，调度器增量读取后通过 SSE 推送到前端。

每个 `step` 事件包含：

| 字段 | 有 tracker 时 | 无 tracker 时 | 说明 |
|------|:---:|:---:|------|
| `loss/current` | ✅ | ❌ | 当前步原始 loss（有毛刺） |
| `loss/average` | ✅ | ❌ | LossRecorder 全量累积平均（天然平滑） |
| `avr_loss` | ❌ | ✅ | 同 loss/average（无 tracker 时的唯一 loss 指标） |
| `lr/group0` | ✅ | ❌ | 参数组 0 的学习率 |
| `global_step` | ✅ | ✅ | 全局步数 |
| `epoch` | ✅ | ✅ | 当前 epoch |

**数据特征**：
- 典型训练（100 图 × 10 epoch × batch_size=1）= ~1000 个 global step
- `log_every_n_steps` 默认 2 → 约 500 个数据点/训练
- `loss/current`（原始 loss）step 级粒度有随机波动（毛刺）
- `avr_loss` / `loss/average` 是全量累积平均，天然去毛刺但后期变化缓慢
- 多轮训练（Train S1 → Train S2）之间 S2 优化器从零重置，loss 可能跳变

### 13.2 数据处理：平滑策略

前端采用 **双层曲线** 显示：

| 曲线 | 数据源 | 视觉 | 用途 |
|------|--------|------|------|
| **原始 loss**（浅色半透明） | `loss/current` 或 `avr_loss` | 浅红 `rgba(231,76,60,0.25)` | 显示真实波动范围 |
| **平滑 loss**（深色实线） | 对原始 loss 做 EMA（α=0.05） | 深红 `#e74c3c` | 清晰展示下降趋势 |
| **LR 曲线** | `lr/group0` | 蓝色 `#3498db` | 监控学习率变化 |

**EMA 平滑算法**（前端计算）：
```
smoothed[0] = raw[0]
smoothed[i] = α * raw[i] + (1 - α) * smoothed[i-1]    // α = 0.05
```

选择 EMA（α=0.05）而非直接使用 `avr_loss` 的原因：
- `avr_loss` 是全量累积平均，后期变化极度缓慢，视觉上几乎一条直线
- EMA 有遗忘因子，能更快反映近期的 loss 趋势变化
- 保留原始 loss（半透明）让用户看到真实波动

### 13.3 多轮训练的关联显示

工作流中的多个 Train 阶段有两种查看模式：

#### 模式 A：单阶段视图（阶段卡片展开时）

显示当前 Train 阶段的独立 loss/lr 曲线，X 轴为该阶段的 step（从 0 开始）：

```
┌─ Train S1 展开详情 ────────────────────────────────────────┐
│  ┌─ Loss ────────────────────────────────────────────────┐  │
│  │  0.8│░░░░░                                           │  │
│  │     │░░░░░\___                                        │  │
│  │  0.4│      ░░░░\____                                  │  │
│  │     │            ░░░░░░░░________                     │  │
│  │  0.1│                    ░░░░░░░░░░░░░                │  │
│  │     └─────────────────────────────────                │  │
│  │      0    100   200   300   400   500  600  step      │  │
│  │      ─ 浅色 = 原始loss  ─ 深色 = EMA平滑              │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─ LR ─────────────────────────────────────────────────┐  │
│  │  4e-4│──────\                                        │  │
│  │  1e-4│             \__________                        │  │
│  │     └─────────────────────────────────                │  │
│  └───────────────────────────────────────────────────────┘  │
│  当前: step 600/1000 · loss 0.123 · lr 1.38e-4 · 1.0s/step │
└──────────────────────────────────────────────────────────────┘
```

#### 模式 B：多阶段对比视图（点击"Loss 对比"按钮）

将多个 Train 阶段的 loss 曲线拼接显示，X 轴为**全局 step**（S1 的 step + S2 的 step 偏移）：

```
┌─ Loss 对比视图 ─────────────────────────────────────────────┐
│                                                              │
│  0.8│░░░░░                                                  │
│     │░░░░░\___                                              │
│  0.4│      ░░░\____                                         │
│     │            ░░░░░░________                             │
│  0.1│                    ░░░░░░░░│▓▓▓▓                      │
│     │                            │▓▓▓▓\___                  │
│ 0.05│                            │    ▓▓▓▓░░░____           │
│     └────────────────────────────│─────────────             │
│      0   100  200  300  400  600 600  650  700  800 1000    │
│      ├──── Train S1 ────────┤├──── Train S2 ──────┤        │
│                              ↑                               │
│                         S2 从 S1 checkpoint 恢复            │
│                         (优化器重置, loss 可能跳变)           │
│                                                              │
│  图例: ─ S1 EMA  ─ S2 EMA  ░ S1 原始  ▓ S2 原始            │
│                                                              │
│  [仅 S1] [仅 S2] [● 全部]  缩放: [1x] [2x] [4x]           │
└──────────────────────────────────────────────────────────────┘
```

**关键设计**：
- 两个 Train 阶段之间用竖线分隔 + 标注衔接关系
- S2 的 loss 可能跳变（因为优化器重置 + 不同数据集），用不同颜色/纹理区分
- 支持过滤显示（仅 S1 / 仅 S2 / 全部）
- 支持缩放（1x / 2x / 4x）查看局部细节

### 13.4 数据存储与降采样

**后端存储**：调度器将所有 step 事件的原数据存储到 `runs/{run_id}/{stage_id}/loss_data.json`，不做任何丢弃。

```json
[
  {"step": 0, "loss": 0.823, "avr_loss": 0.823, "lr": 0.0004, "epoch": 1},
  {"step": 2, "loss": 0.756, "avr_loss": 0.790, "lr": 0.000399, "epoch": 1},
  {"step": 4, "loss": 0.691, "avr_loss": 0.757, "lr": 0.000398, "epoch": 1},
  ...
]
```

**前端降采样**（显示时按需降采样，不丢失数据）：

| 数据点总数 | 降采样策略 |
|-----------|----------|
| ≤ 500 | 不降采样，直接绘制 |
| 500 ~ 2000 | 每 2 点取 1 点（保留 min/max 防止丢失极值） |
| 2000 ~ 10000 | 每 N 点取 1 点（N = 总数 / 1000），保留 min/max |
| > 10000 | LTTB 降采样算法降至 1000 点 |

**降采样保留极值**：无论哪种降采样策略，每个降采样窗口内保留 min 和 max 两个点，确保 loss 的尖峰和谷底不被平滑掉。

### 13.5 实现方式

- 纯 SVG + JavaScript（零外部依赖）
- 图表宽度自适应容器（通常 600-900px）
- 双层曲线：原始（半透明浅色）+ EMA 平滑（深色实线）
- X 轴：step 数，自适应范围，支持鼠标拖拽选择区域放大
- Y 轴：自适应范围，科学计数法标注（如 `4e-4`）
- 鼠标悬停：显示十字线 + 具体数值（step / loss / lr / epoch）
- 多阶段对比：竖线分隔 + 颜色区分 + 过滤控制

---

## 14. 日志查看器

### 14.1 交互设计

```
┌─ 日志 ──────────────────────────────────── [⏸暂停] [🔍搜索] ─┐
│  [10:00:01] ℹ️ 工作流启动，共 4 个阶段                         │
│  [10:00:01] ℹ️ 阶段 preprocess_s1 开始执行                    │
│  [10:00:02] ℹ️ Resizing: 100/100 (100%)                      │
│  [10:00:15] ℹ️ Caching latents: 45/100 (45%)                 │
│  [10:00:43] ✅ 阶段 preprocess_s1 完成 (42s)                  │
│  [10:01:02] ℹ️ 阶段 train_s1 开始执行                         │
│  [10:05:30] ℹ️ epoch 6/10, step 600, loss=0.123, lr=1.38e-4  │
│  [10:05:31] ℹ️ saving checkpoint: anima_lokr-000006.safe...   │
│  ▼ (自动滚动到底部)                                            │
└───────────────────────────────────────────────────────────────┘
```

### 14.2 功能规格

| 功能 | 行为 |
|------|------|
| **自动滚动** | 默认启用，新日志到达时自动滚动到底部 |
| **暂停滚动** | 点击 `[⏸暂停]` 或用户手动向上滚动时自动暂停，显示"已暂停自动滚动"提示 |
| **恢复滚动** | 再次点击或滚动到底部时恢复 |
| **搜索** | 点击 `[🔍搜索]` 弹出搜索栏，支持关键词高亮 + 上/下导航 |
| **阶段过滤** | 下拉选择"全部"/"仅当前阶段"/特定阶段 |
| **日志级别** | 彩色标记：ℹ️ 蓝色信息、⚠️ 黄色警告、❌ 红色错误、✅ 绿色成功 |
| **字体** | monospace 12px |
| **行数限制** | 保留最近 5000 行，超出时删除最早的行 |

---

## 15. 进度控制

### 15.1 控制按钮

| 按钮 | 颜色 | 行为 | 可用条件 |
|------|------|------|---------|
| **▶ 运行** | 绿色 `#27ae60` | 启动工作流调度器 | 空闲时 |
| **■ 停止** | 红色 `#c0392b` | 杀死当前阶段的子进程，中止工作流 | 运行中 |

### 15.2 停止行为

与现有 GUI 一致：**仅支持 Stop（杀死进程），不支持 Pause/Resume。**

1. 用户点击 `[■ 停止]`
2. 调度器发送 SIGTERM 到当前阶段的子进程
3. 等待 5 秒，若未退出则 SIGKILL
4. 标记当前阶段为 `stopped`，后续阶段标记为 `cancelled`
5. 发送 `workflow_end` 事件（status=stopped）
6. 前端更新：当前阶段显示橙色 ■ 已停止，后续阶段恢复为 ⏳ 等待
7. 用户可修改配置后重新点击 `[▶ 运行]`

### 15.3 不支持 Pause 的理由

与现有 GUI 一致：训练过程是有状态的（优化器 momentum、scheduler 状态），Pause 需要保存完整的训练状态，复杂度极高。Stop + 重新运行（通过 checkpointing_epochs 产生的 resumable checkpoint 恢复）是更可靠的方案。

---

## 16. 端到端用户旅程

### 16.1 首次使用

```
1. 启动: python -m workflow
   → pywebview 窗口打开，显示欢迎页

2. 欢迎页:
   ┌───────────────────────────────────────────┐
   │  🔄 Anima Workflow                         │
   │                                            │
   │  创建新工作流                               │
   │  ┌────────────────────────────────────┐   │
   │  │ 名称: [hanechan-lokr-two-stage    ]│   │
   │  │ 位置: [O:\loratool\anima_lora_fork\│   │
   │  │        workflows\                 ]│   │
   │  │              [创建]                │   │
   │  └────────────────────────────────────┘   │
   │                                            │
   │  — 或 —                                    │
   │                                            │
   │  [打开已有工作流]                           │
   │                                            │
   │  最近:                                     │
   │  📁 hanechan-lokr (2 天前)                 │
   │  📁 test-lora (1 周前)                     │
   └───────────────────────────────────────────┘

3. 创建后 → 空的阶段面板 + 基础设施配置提示
```

### 16.2 配置基础设施

```
4. 点击 [⚙ 基础设施设置] → 弹出配置面板
   ├─ 模型路径（留空使用默认）
   ├─ 硬件设置（混合精度、注意力模式）
   └─ [保存]

   → 基础设施配置保存到 workflow.yaml 的 infrastructure 节
```

### 16.3 添加阶段并配置

```
5. 点击 [+ 添加阶段] → Preprocess → 创建 Preprocess S1 卡片

6. 选中 Preprocess S1 → 右侧加载 Schema 表单
   ├─ 数据源: O:\LoRATraining\hanechan
   ├─ Bucket Families: [S1]
   └─ [保存]

7. 点击 [+ 添加阶段] → Train → 创建 Train S1 卡片

8. 选中 Train S1 → 右侧加载方法选择器 + 通用表单
   ├─ 基础类型: [LoKR ▾]
   │   → 加载 LoKR 特定参数
   ├─ 通用参数:
   │   ├─ 学习率: 0.0004
   │   ├─ LR 调度器: cosine
   │   ├─ 最大 Epoch: 10
   │   ├─ 停止 Epoch: 6
   │   ├─ 优化器: CAME
   │   └─ 输入数据集: ☑ Preprocess S1
   ├─ LoKR 参数:
   │   ├─ Network Dim: 16
   │   ├─ Network Alpha: 8
   │   ├─ LoKr Factor: 8
   │   └─ Scale Weight Norms: ⚡1.0 (自动)
   └─ [保存]

9. 类似操作添加 Preprocess S2 (bucket_families=S2) 和 Train S2
   Train S2 额外配置:
   ├─ 输入数据集: ☑ Preprocess S1 + ☑ Preprocess S2
   ├─ Checkpoint 衔接: ○ Train S1 输出
   ├─ 学习率: 0.000138
   ├─ LR 调度器: constant
   └─ 最大 Epoch: 4
```

### 16.4 运行与监控

```
10. 点击 [▶ 运行]
    ├─ 前端验证必选参数
    ├─ 弹出确认: "将执行 4 个阶段的工作流。确认运行？"
    └─ 确认 → 后端创建运行目录 → 调度器启动

11. 监控进度:
    阶段面板实时更新:
    ┌────────────────────────┐
    │ 📁 Preprocess S1    ✅ │  ← 完成 (42s)
    ├────────────────────────┤
    │ 🎯 Train S1       🔄 │  ← 运行中
    │ ▓▓▓▓▓▓▓▓░░░ 60%       │
    │ step 600 · loss 0.123  │
    ├────────────────────────┤
    │ 📁 Preprocess S2    ⏳ │  ← 等待
    ├────────────────────────┤
    │ 🎯 Train S2         ⏳ │  ← 等待
    └────────────────────────┘

    总进度条: ▓▓▓▓▓▓░░░░░░░░░░░░ 35% (阶段 2/4)

12. 展开运行中阶段 → 查看 Loss/LR 曲线 + 详细日志

13. Train S1 到达 epoch 6 → 保存 checkpoint → 阶段结束
    Preprocess S2 自动开始 → 完成
    Train S2 开始 → 使用 Train S1 的 checkpoint → 4 epochs → 完成

14. 所有阶段完成:
    ├─ 总进度: ✅ 100%
    ├─ 各阶段卡片显示 ✅ + 耗时
    └─ 日志显示最终 safetensors 路径
```

### 16.5 查看产物

```
15. 点击 Train S2 卡片 → 展开详情
    ├─ 输出文件: anima_lokr.safetensors
    ├─ 文件大小: 45.2 MB
    ├─ Loss 曲线（4 epochs）
    └─ [打开输出目录] → 在文件管理器中打开
```

### 16.6 重新运行

```
16. 清除上次运行: 点击 [清除运行记录]
    → 删除 runs/ 下所有内容
    → 阶段卡片恢复为 ⏳ 等待状态

17. 修改配置后重新运行
```

---

## 17. 完整 API 更新

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/workflows` | 列出工作流 |
| POST | `/api/workflows` | 创建工作流 |
| GET | `/api/workflows/{name}` | 获取工作流详情 |
| PUT | `/api/workflows/{name}` | 更新工作流 |
| DELETE | `/api/workflows/{name}` | 删除工作流 |
| POST | `/api/workflows/{name}/run` | 运行工作流 |
| POST | `/api/workflows/{name}/stop` | 停止运行 |
| DELETE | `/api/workflows/{name}/runs` | 清除所有运行记录 |
| GET | `/api/workflows/{name}/infrastructure` | 获取基础设施配置 |
| PUT | `/api/workflows/{name}/infrastructure` | 更新基础设施配置 |
| GET | `/api/runs/{run_id}/events` | SSE 流式事件 |
| GET | `/api/runs/{run_id}/log` | 获取日志（支持 ?stage_id=&q= 搜索） |
| GET | `/api/runs/{run_id}/loss-curve` | 获取 loss/lr 数据点（JSON） |
| GET | `/api/schemas/{stage_type}` | 获取参数 Schema |
| GET | `/api/schemas/infrastructure` | 获取基础设施 Schema |
| GET | `/api/schemas/train_common` | 获取通用训练参数 Schema |
| GET | `/api/templates` | 列出配置模板 |
| POST | `/api/import-toml` | 从 TOML 导入配置 |
| GET | `/api/recent-workflows` | 获取最近打开的工作流列表 |
| GET | `/` | WebUI 静态页面 |
