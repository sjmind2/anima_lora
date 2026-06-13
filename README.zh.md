# anima_lora

[English](README.md) | **中文** | [日本語](README.ja.md)

适用于 [Anima](https://huggingface.co/circlestone-labs/Anima) 扩散模型（基于 DiT，flow-matching）的 LoRA / T-LoRA 训练与推理引擎。

本仓库致力于做好以下四件事：

1. **在消费级 GPU 上实现快速 LoRA 训练** — 通过按块 `torch.compile` 针对少量固定形状集合（每个 token 计数族仅编译一个块图）实现端到端加速。
2. **扎实的基础实现** — LoRA、OrthoLoRA 和 T-LoRA 可叠加使用，并无损融合为独立的 DiT 检查点。
3. **为 Anima 工程化实现的最新方法** — Spectrum 推理、DCW 与 SMC-CFG 采样器、OrthoHydraLoRA 以及调制引导，每种方法都针对 Anima 的编译约束进行了端到端实现，而非简单的玩具移植。
4. **广泛的实验性功能** — SPD、ChimeraHydra、Soft Tokens、Turbo 蒸馏、ReFT、IP-Adapter、EasyControl、DirectEdit、嵌入反演。

> 每种方法的**概览图**（DiT 内部结构、LoRA、OrthoLoRA、T-LoRA、HydraLoRA、ReFT、Spectrum、调制、编译优化）位于 [`docs/structure_images/`](docs/structure_images/) — 配合 [`docs/structure/`](docs/structure/) 中的文字说明。

## 最新更新

| 功能 | 描述 | 指南 |
|------|------|------|
| **CAME_C 优化器** | CAME 的融合 CUDA kernel 实现 — 用 8 个自定义 kernel 替代 35+ ATen 操作，速度提升 1.31 倍 | [docs/guidelines/came.zh.md](docs/guidelines/came.zh.md) |
| **LoKR** | 低秩 Kronecker 积适配 — 结构化高秩，自适应参数量 | [docs/guidelines/lokr.zh.md](docs/guidelines/lokr.zh.md) |
| **LoHA** | 低秩 Hadamard 积适配 — 有效秩为 r²，仅需 2× LoRA 参数量 | [docs/guidelines/loha.zh.md](docs/guidelines/loha.zh.md) |
| **CAME 优化器** | 分解式优化器，替代全矩阵二阶矩 — 显著节省显存 | [docs/guidelines/came.zh.md](docs/guidelines/came.zh.md) |
| **分桶族 (Bucket Families)** | 按面积分桶 → 宽高比匹配到 token 计数组，提升编译性能 | [docs/guidelines/bucket-families.zh.md](docs/guidelines/bucket-families.zh.md) |
| **工作流引擎** | WebUI + CLI 多阶段训练流水线，实时进度与 schema 驱动表单 | [docs/guidelines/workflow.zh.md](docs/guidelines/workflow.zh.md) |

---

## 快速开始

### 推荐方式 — Workflow（工作流引擎）

最简单的上手方式是使用内置的 **Workflow 工作流引擎** — 一个 WebUI + CLI 多阶段训练流水线，提供实时进度和表单驱动的配置界面。

```bash
uv sync                   # 安装依赖（Python 3.13）
.venv\Scripts\activate    # Windows: 激活虚拟环境
# source .venv/bin/activate   # Linux/macOS
python -m workflow        # 启动 Workflow WebUI（http://localhost:8765）
```

Workflow WebUI 将引导你完成模型下载、数据集准备、训练和推理 — 全部通过浏览器界面操作。

> 首次使用前，请先进行 HuggingFace 认证：`hf auth login`

Workflow 使用详情请参见 [docs/guidelines/workflow.zh.md](docs/guidelines/workflow.zh.md)。

### 其他入口

<details>
<summary><b>GUI（PySide6 桌面应用）</b></summary>

```bash
make gui                  # 配置编辑器 + 数据集浏览器 + 训练监控
```
</details>

<details>
<summary><b>CLI（make 命令行）</b></summary>

```bash
make preprocess           # VAE 兼容的缩放和验证
make lora                 # 或: PRESET=fast_16gb make lora / PRESET=low_vram make lora / make exp-chimera
make test                 # 使用最新训练的 LoRA 进行采样生成
```
</details>

<details>
<summary><b>一键安装脚本（从发布版安装）</b></summary>

自动安装 [uv](https://astral.sh/uv)（如未安装）、获取最新版本并运行 `uv sync`（无需 git）。安装脚本以带校验和签名的发布资产发布：

```bash
# Linux / macOS
curl -LsSf https://github.com/sjmind2/anima_lora/releases/latest/download/install.sh | sh
```
```powershell
# Windows (PowerShell)
irm https://github.com/sjmind2/anima_lora/releases/latest/download/install.ps1 | iex
```

安装到 `./anima_lora/`（可通过 `ANIMA_DIR` 覆盖）。在 Windows 上还会在桌面创建 **"Anima LoRA GUI"** 快捷方式。

**可复现 / 固定版本安装** — 设置 `ANIMA_VERSION` 来安装特定版本而非最新版：

```bash
ANIMA_VERSION=v1.4.0 sh install.sh       # 或: $env:ANIMA_VERSION='v1.4.0'; irm ... | iex
```

后续可通过 `make update` 原地更新（发布归档合并，无需 git）。想克隆仓库？参见 [安装 → 手动](#手动从克隆安装)。
</details>

---

> **重要提示：训练 LoRA / LyCORIS 时务必使用正则化（reg）数据集。** 不使用 reg 数据进行训练可能导致基础模型能力严重污染 — 适配器可能过拟合于训练图像，导致模型泛化能力下降（如风格渗透、姿势锁定、画质劣化）。请务必包含一组多样化的通用图像作为 reg 数据集，以保持基础模型的原始表现。这对于 Anima 的 DiT 架构尤为关键。

---

## 1. 快速训练

在单张 RTX 5060 Ti 上实现 **13.4 GB 峰值显存 · 1.1 秒/步**，同时进行 **rank=32 的 1MP 分辨率 LoRA 训练** — 通过协同设计数据流水线、注意力机制和编译栈，使 Dynamo 仅看到极少的固定形状集合（每个 token 计数族仅编译一个块图）。

| 优化手段 | 概述 |
|---|---|
| 恒定 token 分桶 | 分桶分为两个 token 计数族 — 4032 和 4200 个 patch — 每个分辨率*精确*填满其计数，因此桶内零填充。前向传播在原生 token 计数下运行，`torch.compile` 仅需为每个不同计数追踪一个块图（共 2 个）。旧的静态填充路径已移除（它将填充泄漏到 flash self-attn 且无法运行此分桶表 — 4200 > 4096）。 |
| 最大填充文本编码器 | 文本输出填充到 512 并填零 — 预训练 DiT 将零值键作为交叉注意力汇聚点，截断会导致生成失败。同时为编译器提供了另一个固定维度。 |
| 按块 `torch.compile` | 每个 DiT 块使用 Inductor 独立编译（`compile_blocks()`）。结合原生 token 分桶，将追踪固定为 2 个块图，消除守卫重编译。 |
| 编译友好的热路径 | 审计了所有前向传播中 dynamo 无法干净追踪的模式 — `einops.rearrange` 替换为显式 `.unflatten()/.permute()` 链，`torch.autocast` 上下文管理器替换为直接 `.to(dtype)` 转换，字典 `.items()` 循环提升出编译区域，FA4 用 `@torch.compiler.disable` 包裹以实现干净的图断点。 |
| Flash Attention 2 | `flash_attn` 2.x，带 SDPA 回退。FA4 已评估并移除 — 参见 [fa4.md](docs/optimizations/fa4.md)。 |

编译流水线详情参见 [docs/optimizations/for_compile.md](docs/optimizations/for_compile.md)。

### 分桶族 — 细节精度与训练速度的权衡

Anima 的 `CONSTANT_TOKEN_BUCKETS` 表将分辨率按 **token 计数 (TC) 族** 分组 — 每个族有固定的 patch 网格面积，因此 `torch.compile` 每个族仅追踪一个块图。较小的 TC 族训练更快（每步 patch 更少）但牺牲细节精度；较大的族保留更多空间信息，代价是计算量和显存增加。

| 族 | Token 计数 | 最大分辨率 | 适用场景 |
|---|---|---|---|
| XS | 1680 | 640×672 | 最快迭代，低细节（面部/姿势实验） |
| S | 2160 | 720×768 | 快速草稿，小数据集 |
| M | 3600 | 960×960 | 质量/速度平衡，适合大多数角色 |
| L | 4032 | 1008×1024 | 高细节，近正方形宽高比 |
| S1 | 1024 | 512×512 | 小正方形，极致低显存 |
| S2 | 4096 | 1024×1024 | 全 1MP 正方形，最大细节 |
| XL | 5040 | 1120×1152 | 最大细节，最宽广高比范围 |

通过配置中的 `bucket_families` 选择（例如 `bucket_families = ["S2"]` 表示仅 1024² 训练）。不同 TC 组的族可以混用 — 每增加一个 TC 族会增加一个编译块图。为获得最佳编译缓存命中率，建议仅使用单个 TC 族。

### 梯度检查点 — 显存与速度的权衡

梯度检查点 (GC) 用计算换取显存：它不在前向时保存所有激活值，而是在反向时重新计算。三个旋钮控制此权衡：

| 设置 | 显存 | 速度 | 适用场景 |
|------|------|------|----------|
| 关闭 GC | 1024² bs=4 约 32 GB | 最快（无重计算） | 24 GB+ 无显存压力的 GPU |
| 全部 GC (`last_n=0`) | 约 8–10 GB | 慢约 1.7 倍（全部 28 个 block 重计算） | 显存紧张，大 batch 或高分辨率 |
| 选择性 GC (`last_n=N`) | 随 N 变化（检查点越少显存越多） | 比全部 GC 快（重计算的 block 更少） | 平衡速度和显存余量 |
| **GC + SAC** | 全部 GC + 约 2–4 GB 额外 | 比标准 GC 快 15–25% | **推荐** 用于启用 GC 的训练 |

**SAC（选择性激活检查点）** 使用 PyTorch 2.12 的 `CheckpointPolicy`，保存昂贵的 attention 运算（flash attention、softmax），反向时仅重计算廉价运算（LayerNorm、逐元素运算、线性投影）。这消除了重计算中最昂贵的部分，同时保持激活内存增加适中。在配置中启用 `gradient_checkpointing_sac = true`（需要 `torch_compile = true`）。

**`last_n` 参数** — 设置为 > 0 时，仅最后 N 个 DiT block（最接近输出层）使用检查点；其余 block 正常运行。由于 `last_n=0` 表示"检查点全部 28 个 block"（最大显存节省），它是最激进的 GC 设置。减小 `last_n`（例如降到 20–22）会逐步以显存换速度，让早期 block 退出检查点。最优值取决于显存预算 — 从较高值（例如 `last_n=22`）开始，仅在遇到 OOM 时增大。

### 性能调优 — GPU 利用率与基线参考

较小的 TC 族会导致高端 GPU 负载不足。例如在 RTX 5090 上，S1 族（TC=1024）在 `bs=4` 时仅能维持 62–69% 的 GPU 利用率 — 计算量太轻，无法充分利用 GPU。此时**增大 batch size** 是最有效的手段：它提高 GPU 利用率而不改变每样本计算量。但更大的 batch 消耗更多显存，因此需要综合平衡 `batch_size` / GC / SAC / `last_n`，在充分利用 GPU 的前提下尽量减少额外损耗。

**性能基线**（RTX 5090，LoKR dim/alpha=16/8 factor=8，CAME_C）：

| 配置 | TC | bs | 秒/步 | 秒/样本 | 说明 |
|------|-----|-----|--------|----------|------|
| S1 | 1024 | 4 | ~0.4 | ~0.10 | 接近该 TC 的理论极限 |
| S2 | 4096 | 1 | ~0.42 | ~0.42 | 计算量线性扩展：4096/1024 ≈ 0.42/0.10 |

S1 在 `bs=4` 时每步处理 4 个样本约 0.4 秒，即 ~0.1 秒/样本 — 基本达到计算下限。S2 在 `bs=1` 时每步处理 1 个样本，token 数为 4 倍，因此每样本时间升至 ~0.42 秒，与 4096/1024 的计算量比例一致。两者均接近各自的性能上限。

**CAME_C 与融合 AdamW** — CAME_C 每步比融合 AdamW 有毫秒级开销，但在 ~0.1 秒/步以上时可忽略不计（不足步耗时的 1%）。

**数值精度** — CAME/CAME_C 在 15k 步后累计误差可能达到 ~1e-5。启用 SAC 会额外产生约 1e-5 的误差。两者均在 LoRA 训练的可接受容差范围内，对最终模型质量的实际影响有限。

---

## 2. 扎实的基础实现

默认训练配置将 **LoRA + OrthoLoRA + T-LoRA** 叠加使用。三者均可通过保存时的 thin-SVD 导出无损融合为独立的 DiT 检查点，因此你可以发布兼容 ComfyUI 的 `*_merged.safetensors`，无需适配器加载器依赖。

| 变体 | 简介 | 详情 |
|---|---|---|
| **LoRA** | 经典低秩适配，rank 16–32。 | — |
| **OrthoLoRA** | 基于 SVD 参数化并施加正交性正则化；导出为标准 LoRA。 | [psoft-integrated-ortholora.md](docs/methods/psoft-integrated-ortholora.md) |
| **T-LoRA** | 时间步相关的秩遮罩 — 高噪声时低秩，低噪声时全秩。训练时使用遮罩，融合后位等价。 | [timestep_mask.md](docs/methods/timestep_mask.md) |

**对比图** — 相同提示词，`er_sde` 30 步，`cfg=4.0`，1024²。每个 LoRA 以 rank 16 训练 2 个 epoch，使用 20% 子集，训练种子 42；推理种子 `{41, 42, 43}`。可使用 `python _archive/bench_methods.py` 复现。

|  | **LoRA** | **OrthoLoRA + T-LoRA** |
|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/lora/20260423-154854-014_41_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155545-258_41_.png" width="320"> |
| seed 42 | <img src="docs/side_by_side/lora/20260423-154938-584_42_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155631-762_42_.png" width="320"> |
| seed 43 | <img src="docs/side_by_side/lora/20260423-155024-080_43_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155718-280_43_.png" width="320"> |

<details>
<summary>基础模型及各变体（标准、OrthoLoRA、T-LoRA）</summary>

|  | **标准（基础）** | **OrthoLoRA** | **T-LoRA** |
|:---:|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/plain/20260423-160513-382_41_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155109-338_41_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155327-834_41_.png" width="240"> |
| seed 42 | <img src="docs/side_by_side/plain/20260423-160556-697_42_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155155-526_42_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155413-304_42_.png" width="240"> |
| seed 43 | <img src="docs/side_by_side/plain/20260423-160640-759_43_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155241-905_43_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155458-996_43_.png" width="240"> |

</details>

**融合**：

```bash
make merge                                  # 以 multiplier 1.0 融合最新 LoRA
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8
```

默认拒绝非线性增量变体（ReFT / HydraLoRA `_moe`）；`--allow-partial` 会跳过这些，仅融合 LoRA 部分。

---

## 3. 为 Anima 工程化实现的最新方法

精选五篇近期论文，针对 Anima 端到端实现，并配备使其真正可用所需的工程优化 — 而非玩具式重实现。

| 方法 | 简介 | 工程备注 | 文档 |
|---|---|---|---|
| **Spectrum 推理** | 基于 Chebyshev 多项式特征预测的训练无关加速（Han 等，CVPR 2026）— 默认设置下约 1.75×，更激进的调度可达约 5×（质量有取舍）。缓存步骤中每个 transformer 块都被跳过 — 仅运行 `t_embedder` + `final_layer` + `unpatchify`。 | 在 `final_layer` 上使用 `register_forward_pre_hook` 捕获块输出，无需 monkey-patch 模型；自适应窗口调度将真实前向集中在早期高噪声步。稳定的 ComfyUI 节点位于独立仓库：[ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)。 | [spectrum.md](docs/methods/spectrum.md) |
| **DCW 校准器** | 采样器级 SNR-t 偏差校正（Yu 等，CVPR 2026）— 沿 LL Haar 频带将每个 Euler 步的 `prev_sample` 混合向模型的 `x0_pred`。两种模式：标量 `λ`（离线调优）和 **v4 可学习** 逐提示校准器（带在线观测）。 | v4 头以 `(宽高比, 提示词, 观测前缀间隙)` 为条件，在 `k=7` 热身步后启动。偏差方向在 Anima 上表现为 **(CFG × 宽高比) 相关** — 论文方向在 CFG=4 非正方形时，论文反方向在 CFG=1 / 1024² 时。通过 `make dcw` 逐检查点训练。 | [dcw.md](docs/methods/dcw.md) |
| **SMC-CFG** | 训练无关的滑模 CFG 校正（速度空间）（Wang 等，CFG-Ctrl）— 将条件/无条件组合视为控制问题，应用于残差 `e = v_cond − v_uncond`。无需额外 DiT 前向。 | 默认使用 **α 自适应变体**：论文的固定增益 `k`（在 Anima CFG=4 下偏差约 14 倍，可见振铃）被替换为每步 `k_t = α·mean(\|e_t\|)`。`make test-smc-cfg`（λ=5，α=0.2）；可与 Spectrum 和调制引导组合。 | [smc_cfg.md](docs/methods/smc_cfg.md) |
| **OrthoHydraLoRA** | MoE 风格多头 LoRA，带正交化专家和逐层路由 — 共享 `lora_down`，逐专家 `lora_up_i`，可学习的逐样本路由器。面向多风格训练，避免单一低秩子空间产生的跨风格串扰。原始论文：[arXiv:2605.03252](https://arxiv.org/abs/2605.03252)。 | 保存两个并列文件：`anima_hydra.safetensors`（融合后的 LoRA，ComfyUI 即插即用）和 `anima_hydra_moe.safetensors`（完整多头）。ComfyUI 中通过内置的 **Anima Adapter Loader** 节点（`custom_nodes/comfyui-hydralora/`）实现实时路由，该节点安装了重现 `HydraLoRAModule.forward` 的逐 Linear 前向钩子。 | [hydra-lora.md](docs/methods/hydra-lora.md) |
| **调制引导** | 蒸馏一个 `pooled_text_proj` MLP，将 AdaLN 调制系数引向质量正向方向（Starodubcev 等，ICLR 2026）。教师看到真实交叉注意力；学生看到归零的交叉注意力但通过调制接收池化文本。 | 通过 `make distill-mod` 针对冻结的 DiT 训练。推理时在 AdaLN 时刻应用投影，因此可与任何 LoRA 变体组合；`make test MOD=1` 运行启用它的采样（可与 `SPECTRUM=1` 组合）。 | [mod-guidance.md](docs/methods/mod-guidance.md) |

---

## 4. 实验性功能

每个功能都配有文档 — 参见链接了解用法、标志和注意事项。

| 功能 | 简介 | 文档 |
|---|---|---|
| **SPD** | Spectral Progressive Diffusion（Xiao 等，2026）— 训练无关的多分辨率推理（`--spd`）：在低分辨率下运行早期噪声主导步骤，然后通过频谱噪声扩展注入高频细节。可选轨迹适配器微调（`make exp-spd`）。 | [spd.md](docs/experimental/spd.md) |
| **ChimeraHydra** | 双池加性 MoE：内容池（逐层路由器）加频率池（基于 FEI + σ 特征的网络路由器），各自在互不相交的 SVD 子空间上的非对称 HydraLoRA。融合了 HydraLoRA + TimeStep Master + FeRA。`make exp-chimera`。 | [chimera-hydra.md](docs/experimental/chimera-hydra.md) |
| **Soft Tokens** | SoftREPA（Lee 等，NeurIPS 2025）— 逐层 × 逐 t 可学习文本 token（约 1M 参数）拼接到 `crossattn_emb` 中；DiT 冻结。`make exp-soft-tokens`。 | [soft_tokens.md](docs/experimental/soft_tokens.md) |
| **Turbo** | 将 28 步教师模型的 Decoupled DMD 蒸馏（Liu 等，2025）为 4–8 步生成器。输出为标准 LoRA — 使用 `--infer_steps 4 --cfg 1.0` 推理。`make exp-turbo`。 | [turbo_anima_dmd_lora.md](docs/proposal/turbo_anima_dmd_lora.md) |
| **DirectEdit** | Flow-inversion 图像编辑（Yang & Ye，2026）— 反演到噪声，交换编辑条件，用 V-injection 重新去噪。源描述来自 **Anima Tagger**（图像 → Anima 格式标签）。`make exp-test-directedit`。 | [directedit_editing_v3.md](docs/experimental/directedit_editing_v3.md) |
| **ReFT** | 块级残差流干预（LoReFT，NeurIPS 2024）。可与任何 LoRA 变体组合。 | [reft.md](docs/methods/reft.md) |
| **IP-Adapter** | 解耦图像交叉注意力（Ye 等，2023）。DiT 冻结；训练 Perceiver 重采样器 + 逐块 `to_k_ip`/`to_v_ip`。 | [ip-adapter.md](docs/experimental/ip-adapter.md) |
| **EasyControl** | 扩展自注意力图像条件。DiT 冻结；在自注意力 + FFN 上训练逐块条件 LoRA + 标量 `b_cond` 门控。 | [easycontrol.md](docs/experimental/easycontrol.md) |
| **嵌入反演** | 通过冻结的 DiT 优化文本嵌入以匹配目标图像。 | [invert.md](docs/methods/invert.md) |

> **想要贡献？** 有两个领域外部帮助将产生巨大影响：**IP-Adapter 生产化**（测试、公开参考检查点、更轻量的视觉编码器）和 **EasyControl 适配器**（canny / depth / pose / … — 每种控制类型是一个独立的 PR）。参见 [CONTRIBUTING.md → Priority areas](CONTRIBUTING.md#priority-areas)。

---

## 安装

> 快速一行安装见上方 [快速开始](#快速开始)。以下是手动克隆安装路径。

### 手动从克隆安装

```bash
uv sync                   # Python 3.13，预构建 flash attention 2
hf auth login
make download-models      # 下载 DiT + Qwen3 TE + QwenImage VAE（+ SAM3 / MIT / PE，用于遮罩和图像条件）到 models/
# 将训练图片放入 image_dataset/，附带 .txt 描述文件
make gui                  # 推荐 — 配置编辑器 + 数据集浏览器 + 训练监控
```

`uv sync` 解析到 **torch 2.12 + CUDA 13.2**。

> **Anima 以 uv 锁定的应用环境发布，而非通用 pip 包。** `pyproject.toml` 锁定 `python ==3.13.*`、特定的 torch / flash-attn wheel URL 和 `index-strategy = "unsafe-best-match"` — 这些是维护者选定的已知稳定版本。使用 `uv sync` 对已提交的 `uv.lock` 进行安装；不要通过 `pip install` 从 `pyproject.toml` 安装（pip 不会遵守 uv 的索引策略或预构建 flash-attn wheel）。

命令行路径：

```bash
make preprocess           # VAE 兼容的缩放和验证
make lora                 # 或: PRESET=fast_16gb make lora / PRESET=low_vram make lora / make exp-chimera
make test                 # 使用最新训练的 LoRA 进行采样生成
```

配置链：`configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI 参数`。可通过 `PRESET=low_vram make lora` 或 `--network_dim 32 --max_train_epochs 64` 覆盖。完整标志参考见 [docs/guidelines/training.md](docs/guidelines/training.md) 和 [docs/guidelines/inference.md](docs/guidelines/inference.md)。

---

## 文档

| 文档 | 内容 |
|------|------|
| [guidelines/training.md](docs/guidelines/training.md) | 训练标志、LoRA 变体、描述洗牌、遮罩损失、数据集配置 |
| [guidelines/inference.md](docs/guidelines/inference.md) | 推理标志、P-GRAFT、提示文件、LoRA 格式转换 |
| [optimizations/](docs/optimizations/) | 编译流水线、FA4 复盘、CUDA 13.2 |
| [methods/](docs/methods/) | 每种方法一篇文档 — HydraLoRA、ReFT、Spectrum、反演、调制引导、T-LoRA、OrthoLoRA |

---

## 许可证

工具包代码：[MIT](LICENSE)。

Anima / CircleStone **基础模型权重** 以 **CircleStone Labs Non-Commercial License v1.0** 发布，不由本仓库再许可。从这些权重训练的任何 LoRA、微调或融合检查点均为衍生作品，继承非商业条款。参见 [NOTICE](NOTICE)。
