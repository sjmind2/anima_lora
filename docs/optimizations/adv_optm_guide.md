# Advanced Optimizers (Adopt_Adv / Prodigy_Adv) 使用指南

本文档说明如何在 Anima LoRA 训练管道中使用 `adv_optm` 包提供的 `Adopt_Adv` 和 `Prodigy_Adv` 优化器。

## 概述

[Advanced Optimizers (AIO)](https://github.com/Koratahiu/Advanced_Optimizers) 是一个高性能深度学习优化器集合，本项目已集成 `adv_optm==2.2.3` 版本。

| 优化器 | 说明 | 适用场景 |
|--------|------|----------|
| `Adopt_Adv` | Adam 变体，支持独立 beta2，稳定性更好 | 小批量训练、需要稳定收敛的场景 |
| `Prodigy_Adv` | 基于 D-Adaptation 的自动学习率调整 | 不确定最佳学习率时，自动调节 lr |

### 安装

依赖已在 `pyproject.toml` 中声明，运行 `uv sync` 即可安装。手动安装：

```bash
uv pip install adv_optm==2.2.3
```

### 与现有优化器的关系

| 优化器 | 来源 | 默认 |
|--------|------|------|
| `AdamW` | `torch.optim` | ✅ 项目默认，fused=True |
| `Prodigy` | `prodigyopt` | — |
| `Adopt_Adv` | `adv_optm` | — |
| `Prodigy_Adv` | `adv_optm` | — |

---

## 脚本 + 配置方式

### Adopt_Adv 配置示例

在 TOML 配置文件（如 `configs/base.toml` 或 gui-methods variant 文件）中设置：

```toml
optimizer_type = "Adopt_Adv"
learning_rate = 2e-5

# 通过 optimizer_args 传递 adv_optm 特有参数
optimizer_args = ["atan2=True", "stochastic_rounding=True", "weight_decay=0.01"]
```

#### Adopt_Adv 可用特性参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atan2` | bool | False | 用 atan2 替代 eps，自动裁剪更新到 [-2, 2]。**强烈推荐启用**，Adopt_Adv 不加 atan2 容易不稳定 |
| `stochastic_rounding` | bool | False | BF16 随机舍入，保留小梯度更新。BF16 训练推荐启用 |
| `factored` | bool | False | SMMF 因式分解模式，大幅减少内存但增加 ~18% 计算开销 |
| `ademamix` | bool | False | 双 EMA 系统，保留长期梯度记忆。增加 1 个额外状态 |
| `simplified_ademamix` | bool | False | 累加器动量（单 EMA 变体），无额外内存开销 |
| `orthograd` | bool | False | 移除与权重平行的梯度分量，减少过拟合。+33% 时间开销 |
| `cautious` | bool | False | 仅在梯度方向与动量方向一致时应用更新。无额外开销 |
| `grams` | bool | False | 纯梯度方向更新。与 cautious 互斥（grams 优先）。无额外开销 |
| `kbeta` | bool | False | Kourkoutas-β 层级自适应 beta2。适用于噪声/高 lr 训练 |
| `fused_backward_pass` | bool | False | 融合反向传播，减少峰值内存 |
| `compiled_optimizer` | bool | False | 启用 torch.compile 优化 optimizer step |

#### 推荐配置

**基础稳定配置**（推荐起步）：
```toml
optimizer_type = "Adopt_Adv"
learning_rate = 2e-5
optimizer_args = ["atan2=True"]
```

**内存优化配置**（低 VRAM 场景）：
```toml
optimizer_type = "Adopt_Adv"
learning_rate = 2e-5
optimizer_args = ["atan2=True", "factored=True"]
```

**长期训练配置**（大步数训练）：
```toml
optimizer_type = "Adopt_Adv"
learning_rate = 2e-5
optimizer_args = ["atan2=True", "ademamix=True", "beta3=0.9999", "alpha=5"]
```

### Prodigy_Adv 配置示例

```toml
optimizer_type = "Prodigy_Adv"
learning_rate = 1.0

# D-Adaptation 会自动调整实际学习率
optimizer_args = ["weight_decay=0.01"]
```

#### Prodigy_Adv 可用特性参数

与 Adopt_Adv 共享大部分特性（atan2、stochastic_rounding、factored、orthograd、cautious、grams、kbeta 等），额外参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d0` | float | 1e-6 | D-Adaptation 初始估计值 |
| `growth_rate` | float | inf | D 的增长率限制 |
| `ds` | float | 0.0 | D 的初始值（0 = 自动） |

#### 推荐配置

**标准配置**：
```toml
optimizer_type = "Prodigy_Adv"
learning_rate = 1.0
optimizer_args = ["weight_decay=0.01"]
```

**稳定配置**（不确定学习率时）：
```toml
optimizer_type = "Prodigy_Adv"
learning_rate = 1.0
optimizer_args = ["atan2=True", "stochastic_rounding=True"]
```

#### ⚠️ 重要注意事项

- **学习率必须设为 ~1.0**：Prodigy_Adv 使用 D-Adaptation 自动调整实际学习率。设置 `learning_rate = 1.0`（而不是通常的 1e-5），系统会在日志中警告 lr 过低
- **不适用于多 LR 组**：当 `unet_lr` / `text_encoder_lr` 分别设置时，只有第一个 lr 生效

---

## GUI 使用方式

### 方法一：通过 variant 文件设置

1. 在 GUI 中选择 Method 和 Variant
2. 在配置表单中找到 `optimizer_type` 字段
3. 将值改为 `Adopt_Adv` 或 `Prodigy_Adv`
4. 点击 **Save** 保存

### 方法二：通过 Extra args 传递 optimizer_args

GUI 的 Extra args 功能允许传递 TOML 格式的额外参数：

1. 点击配置表单下方的 **Extra args** 按钮展开文本框
2. 输入 TOML 格式的参数：

```toml
optimizer_args = ["atan2=True", "stochastic_rounding=True"]
```

3. 点击 **Save**，参数会被合并到 variant 文件中
4. 如果需要修改 optimizer_type，直接在表单字段中输入 `Adopt_Adv` 或 `Prodigy_Adv`

### 完整 GUI 操作示例（Adopt_Adv）

1. 启动 GUI：`python -m gui`
2. 选择 Method → `lora`，Variant → `LoRA`
3. 在表单中修改以下字段：
   - `optimizer_type` → 输入 `Adopt_Adv`
   - `learning_rate` → 保持 `2e-5`
4. 展开 **Extra args**，输入：
   ```toml
   optimizer_args = ["atan2=True"]
   ```
5. 点击 **Save**
6. 运行 **Preprocess** → **Train**

### 完整 GUI 操作示例（Prodigy_Adv）

1. 选择 Method → `lora`，Variant → `LoRA`
2. 修改字段：
   - `optimizer_type` → 输入 `Prodigy_Adv`
   - `learning_rate` → 改为 `1.0`
3. 展开 **Extra args**，输入：
   ```toml
   optimizer_args = ["weight_decay=0.01"]
   ```
4. **Save** → **Preprocess** → **Train**

---

## 性能损耗说明

以下数据基于 adv_optm 官方 SDXL (6.5GB) 基准测试，与项目默认的 `AdamW (fused=True)` 进行对比。

### 内存开销

| 配置 | 每参数状态内存 | 相对 AdamW | 说明 |
|------|---------------|------------|------|
| AdamW (fused) | 8 bytes (fp32 m + v) | 基准 | 项目默认 |
| Adopt_Adv / Prodigy_Adv 基础模式 | 8 bytes | **持平** | 无额外内存 |
| + Factored | ~2 bytes | **减少 ~75%** | 4 个小向量 + 1-bit 符号状态 |
| + AdEMAMix | ~12-16 bytes | **增加 50-100%** | 额外慢速 EMA 状态 |
| + Simplified_AdEMAMix | 8 bytes | **持平** | 累加器替代标准 EMA |

以 LoRA dim=32 (~30M 可训练参数) 为例：

| 配置 | 额外内存 |
|------|----------|
| AdamW (fused) | ~240 MB |
| Adopt_Adv 基础 | ~240 MB（持平） |
| Adopt_Adv + Factored | ~60 MB（减少 ~180 MB） |
| Adopt_Adv + AdEMAMix | ~360-480 MB（增加 ~120-240 MB） |

### 计算开销（步长时间）

| 特性 | 额外步长时间 | 说明 |
|------|-------------|------|
| 基础模式（无额外特性） | **+5~15%** | 无 fused kernel 融合，但差异不大 |
| Factored | +18% | SMMF 分解/重构循环 |
| Factored + AdEMAMix | +41% | 3 个因式分解状态 |
| OrthoGrad | +33% (BS=4) | 大 batch size 时影响递减 |
| Stochastic Rounding | <5% | 几乎无感 |
| Cautious / Grams / atan2 | **0%** | 纯数学操作，无额外 kernel |
| Kourkoutas-β | **0%** | 仅调整 beta2 标量 |
| torch.compile (compiled_optimizer=True) | 首次编译慢，后续可抵消 5-10% | 需要稳定的计算图 |

### 总体评估

| 使用场景 | 推荐配置 | 预期性能 |
|----------|----------|----------|
| 日常训练，追求稳定 | Adopt_Adv + atan2 | 比 AdamW 慢 ~5-10%，内存持平 |
| 低 VRAM 环境 | Adopt_Adv + atan2 + Factored | 比 AdamW 慢 ~20-25%，内存减少 ~75% |
| 不确定最佳 lr | Prodigy_Adv (lr=1.0) | 比 AdamW 慢 ~5-10%，内存持平 |
| 长期训练 + 小批量 | Adopt_Adv + atan2 + AdEMAMix | 比 AdamW 慢 ~25-35%，内存增加 ~50-100% |

**核心结论**：
- Adopt_Adv / Prodigy_Adv 基础模式的开销很小（5-15%），对大多数用户可以接受
- 内存敏感场景建议启用 Factored，以 ~20% 的速度换取 ~75% 的内存节省
- atan2、Cautious、Grams 等特性是"免费的午餐"——无额外开销即可提升训练稳定性
- OrthoGrad 的 +33% 开销较显著，仅在 full fine-tuning 且无 weight decay 时考虑

---

## 参考链接

- [Advanced Optimizers GitHub](https://github.com/Koratahiu/Advanced_Optimizers)
- [SMMF 论文](https://arxiv.org/abs/2412.08894)（Factored 模式）
- [AdEMAMix 论文](https://arxiv.org/abs/2409.03137)
- [adam-atan2](https://github.com/lucidrains/adam-atan2-pytorch)（atan2 特性）
- [Kourkoutas-β 论文](https://arxiv.org/abs/2508.12996)
- [C-Optim (Cautious)](https://github.com/kyleliang919/C-Optim)
- [Grams](https://github.com/Gunale0926/Grams)
