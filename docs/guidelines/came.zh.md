[English](came.md) | **中文**

# CAME 优化器 — 置信度引导的自适应矩阵评估

CAME 是一种分解式优化器，用行矩和列矩（`exp_avg_sq_row`、`exp_avg_sq_col`）替代全矩阵二阶矩（`exp_avg_sq`），大幅降低 2D 及以上参数的优化器状态显存占用。它还引入了残差校正机制，以补偿分解近似带来的误差。

CAME 是 [工作流引擎](workflow.zh.md) 中的默认优化器，特别适合 LyCORIS 变体训练（LoKR、LoHA），这类训练中分解模块产生大量小型参数矩阵。

## 用法

### CLI / TOML 配置

在方法或预设 TOML 中设置 `optimizer_type`：

```toml
optimizer_type = "CAME"
learning_rate = 1.5e-5
```

通过 `optimizer_args` 传递优化器专用参数：

```toml
optimizer_type = "CAME"
optimizer_args = ["weight_decay=0.01", "betas=0.9,0.999,0.9999"]
learning_rate = 1.5e-5
```

### 工作流

CAME 是工作流模式（`workflow/schemas/train_common.yaml`）中的默认优化器。无需额外配置——除非主动更改，否则会自动选用。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `betas` | (0.9, 0.999, 0.9999) | （beta0 用于 exp_avg，beta1 用于二阶矩，beta2 用于残差矩） |
| `eps` | (1e-30, 1e-16) | （eps0 用于更新稳定性，eps1 用于残差稳定性） |
| `clip_threshold` | 1.0 | RMS 裁剪阈值——防止更新爆炸 |
| `weight_decay` | 0.0 | 权重衰减系数 |
| `lr` | — | 学习率。**建议：设为等效 AdamW 学习率的 0.5–0.9 倍。** |

### 内部行为

- **2D+ 参数**：使用行/列二阶矩 + 残差校正的分解更新。相同形状的参数会被堆叠为批量张量以提高效率。
- **1D 参数**（偏置、标量）：非分解的 Adam 风格更新，使用完整的 `exp_avg_sq`。

## 与其他优化器的对比

| 优化器 | 显存（每参数） | 自动学习率 | 最适用场景 | 来源 |
|--------|---------------|-----------|-----------|------|
| **AdamW** | 2 × 完整状态 | 否 | 通用场景，默认选择 | `torch.optim` |
| **CAME** | ~2 × (行+列) 状态 | 否 | LyCORIS 变体，显存受限场景 | 内置 (`library/training/came_optimizer.py`) |
| **Adopt_Adv** | 可配置（含分解选项） | 否 | 小批量稳定性，长训练 | `adv_optm` 包 |
| **Prodigy_Adv** | 2 × 完整状态 + D-Adaptation 状态 | **是**（设 `lr=1.0`） | 不确定最优学习率时 | `adv_optm` 包 |

完整的高级优化器指南（Adopt_Adv、Prodigy_Adv），请参阅 [docs/optimizations/adv_optm_guide.md](../optimizations/adv_optm_guide.md)。

## 推荐学习率

CAME 的分解二阶矩估计与 AdamW 的完整估计行为不同。实践经验：

| 场景 | AdamW 学习率 | CAME 学习率 |
|------|-------------|------------|
| LoRA rank 32 | 2e-4 | 1–1.5e-4 |
| LoKR dim 8 | 2e-5 | 1.5e-5 |
| LoHA dim 32 | 1e-4 | 7e-5 |

经验法则：**以 AdamW 学习率的 0.7 倍作为起点**，再行调整。

## 已知限制

- **1D 参数**使用非分解（Adam 风格）更新——偏置和标量无显存节省。
- **torch.compile 分组堆叠**将相同形状的参数分组为批量张量。当存在大量不同形状时，分组开销可能削弱收益。实际中极少遇到此问题。
- **不兼容**大于 1 的梯度累积因子（需注意）——每步状态更新本身正确，但等效学习率缩放可能与 AdamW 不同。
