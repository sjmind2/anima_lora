# sjmind2/anima_lora Fork 贡献总结

## 概述

| 项目 | 信息 |
|------|------|
| Fork 仓库 | sjmind2/anima_lora |
| 上游仓库 | lora-tool/anima_lora |
| BASE Tag | `4aefb84`（2026-05-17） |
| 统计周期 | 2026-05-17 ~ 2026-05-24 |
| 已合并 PR | 9 个 |
| 未合并分支 | 4 个 |
| 涉及文件 | 40+ 个 |
| 总代码变更 | +3500 / -200（约） |

---

## 一、已合并贡献

### 直接提交到 main（非 PR）

#### 高级优化器支持

| 字段 | 内容 |
|------|------|
| Commit | `40a732d` |
| 日期 | 2026-05-17 |
| 作者 | sjmind2 |
| 变更量 | 5 files, +320/-22 |
| Tier | Tier 1.5（效率改进） |

**变更内容：**
- 新增高级优化器支持（在 `library/training/optimizers.py` 中实现）
- 添加 REFT 方法配置 `configs/gui-methods/reft.toml`
- 新增优化器使用指南 `docs/optimizations/adv_optm_guide.md`
- 更新 GUI 说明模块 `gui/explanations/__init__.py`
- 更新依赖配置 `pyproject.toml`

---

### PR #1 — feat-mutiple-datasets

#### 新增数据集子集扫描与管理功能

| 字段 | 内容 |
|------|------|
| Commit | `ce42df2` |
| 日期 | 2026-05-18 |
| 作者 | sjmind2 |
| 变更量 | 12 files, +1225/-155 |
| Tier | Tier 1.5（功能增强） |

**变更内容：**
- 新增 `scan_source_dir` 函数，支持从源目录自动扫描生成子集配置
- 为 GUI 配置界面添加子集管理面板（`gui/config_tab.py` +186 行）
- 扩展预处理脚本支持树状目录处理
- 新增 `resize_images`、`cache_latents`、`cache_text_embeddings` 的 tree 模式
- 重构数据集配置合并逻辑
- 添加单元测试（`test_preprocess_tree.py`、`test_scan_source_dir.py`）
- 补充多语言翻译词条

**涉及文件：**
- `gui/__init__.py`（+63）
- `gui/config_tab.py`（+186）
- `gui/i18n.py`（+20）
- `library/config/io.py`（+19/-）
- `library/datasets/base.py`（+29/-）
- `preprocess/cache_latents.py`（+260/-）
- `preprocess/cache_text_embeddings.py`（+160/-）
- `preprocess/resize_images.py`（+114）
- `scripts/tasks/_common.py`（+64）
- `scripts/tasks/preprocess.py`（+210/-）
- `tests/test_preprocess_tree.py`（+76）
- `tests/test_scan_source_dir.py`（+179）

---

### PR #2 — feat-mutiple-datasets

#### 修复预处理子进程 caption 配置传递

| 字段 | 内容 |
|------|------|
| Commit | `f7f43d0` |
| 日期 | 2026-05-18 |
| 作者 | sjmind2 |
| Tier | Tier 1（Bug Fix） |

**变更内容：**
- 修复 `config_tab` 中预处理子进程未正确传递 caption 相关配置参数的问题

---

### PR #3 — feat-mutiple-datasets

#### 新增 XY 参数绘图套件，优化适配器模块热重载

| 字段 | 内容 |
|------|------|
| Commit | `32efa08` |
| 日期 | 2026-05-19 |
| 作者 | sjmind2 |
| Tier | Tier 1.5（新功能） |

**变更内容：**
- 新增 XY 参数绘图完整套件
- 新增文件：`grid.py`、`xyplot.py`、`xy_inputs.py`、`xyplot_widgets.js`
- 优化适配器模块热重载机制
- 修改 `custom_nodes/comfyui-hydralora/__init__.py` 和 `adapter.py`

---

### PR #4 — feat-mutiple-datasets

#### XY Plot LoRA 输入支持与 Hook 补丁修复

| 字段 | 内容 |
|------|------|
| Commit | `52d11d1` |
| 日期 | 2026-05-19 |
| 作者 | sjmind2 |
| Tier | Tier 1.5 |

**变更内容：**
- 为 XY Plot 添加 LoRA XY 输入支持
- 修复 hook patch 逻辑

---

### PR #5 — feat-mutiple-datasets

#### 修复 XY Plot 动态控件逻辑

| 字段 | 内容 |
|------|------|
| Commit | `bb35c70` |
| 日期 | 2026-05-19 |
| 作者 | sjmind2 |
| Tier | Tier 1（Bug Fix） |

**变更内容：**
- 修复 `xyplot_widgets` 中动态显示控件逻辑
- 优化节点注册方式

---

### PR #6 — feat-xy-plot

#### XY Plot 完善

| 字段 | 内容 |
|------|------|
| Commits | `1495ce1`、`8d08c45` |
| 日期 | 2026-05-19 |
| 作者 | sjmind2 |
| Tier | Tier 1.5 |

**变更内容：**
- 添加 LoRA XY 输入支持并修复 hook patch 逻辑
- 完善 XY Plot 功能
- 涉及文件：`custom_nodes/comfyui-hydralora/__init__.py`、`adapter.py`、`grid.py`（新增）、`xyplot_widgets.js`

---

### PR #7 — feat-lycoris

#### 新增 LOCON/LOHA/LOKR 三种 LyCORIS 训练方法

| 字段 | 内容 |
|------|------|
| Commit | `8b16d4d` |
| 日期 | 2026-05-22 |
| 作者 | sjmind2 |
| Tier | Tier 2（新 LoRA Adapter 方法） |

**变更内容：**
- 新增 LOCON 模块 `networks/lora_modules/locon.py`（216 行）
- 新增 LOHA 模块 `networks/lora_modules/loha.py`（253 行）
- 新增 LOKR 模块 `networks/lora_modules/lokr.py`（552 行）
- 新增 LyCORIS 公共函数库 `networks/lora_modules/lycoris_functional.py`（113 行）
- 新增模型保存模块 `networks/lora_save.py`（194 行）
- 新增工具函数 `networks/lora_utils.py`（86 行）
- 更新网络工厂、配置、加载等核心模块
- 新增配置文件：`configs/methods/locon.toml`、`loha.toml`、`lokr.toml`
- 新增 GUI 配置：`configs/gui-methods/locon.toml`（273 行）、`loha.toml`（77 行）、`lokr.toml`（80 行）
- 新增 GUI 说明文档（中英文 HTML）

---

### PR #8 — feat-lycoris

#### 多 LoRA 系列训练配置与功能优化

| 字段 | 内容 |
|------|------|
| Commit | `ebdd6d8` |
| 日期 | 2026-05-23 |
| 作者 | sjmind2 |
| Tier | Tier 1.5 |

**变更内容：**
- 新增多 LoRA 系列（LoRA/LoCON/LOHA/LOKR）训练配置
- 功能整合与优化

---

### PR #9 — feat-performance

#### 性能优化

| 字段 | 内容 |
|------|------|
| Commit | `01ebe96` |
| 日期 | 2026-05-24 |
| 作者 | sjmind2 |
| Tier | Tier 1.5（性能优化） |

**变更内容：**
- 优化训练循环性能 `library/training/loop.py`（+312/-）
- 优化 LOHA、LOKR、LyCORIS functional 模块
- 优化模型保存逻辑
- 更新多种方法的 GUI 配置

**涉及文件：**
- `configs/gui-methods/loha.toml`
- `configs/gui-methods/lokr.toml`
- `configs/gui-methods/lora.toml`
- `gui/__init__.py`
- `library/training/loop.py`
- `networks/lora_modules/loha.py`
- `networks/lora_modules/lokr.py`
- `networks/lora_modules/lycoris_functional.py`
- `networks/lora_save.py`

---

## 二、未合并分支

| 分支 | 最新 Commit | 状态 | 说明 |
|------|-------------|------|------|
| `feat-adv-optm` | `5a39e0e` | behind 4 commits | 高级优化器支持（已部分合入 main） |
| `feat-xy` | `6fe5e8d` | 未合并 | XY 相关改动 |
| `feat-lycoris` | `ebdd6d8` | 已通过 PR #8 合并 | — |
| `feat-mutiple-datasets` | `bb35c70` | 已通过 PR #5 合并 | — |
| `feat-xy-plot` | `8d08c45` | 已通过 PR #6 合并 | — |
| `feat-performance` | `01ebe96` | 已通过 PR #9 合并 | — |

---

## 三、总体统计汇总

### 按 Tier 分类

| Tier | 数量 | 说明 |
|------|------|------|
| Tier 1 | 2 | Bug Fix（PR #2、PR #5） |
| Tier 1.5 | 7 | 功能增强 / 效率改进（直接提交、PR #1、#3、#4、#6、#8、#9） |
| Tier 2 | 1 | 新 LoRA Adapter 方法（PR #7） |

### 按功能领域分类

| 领域 | PR | 说明 |
|------|-----|------|
| 优化器 | 直接提交、PR #9 | 高级优化器支持、性能优化 |
| 数据集管理 | PR #1、#2 | 子集扫描、预处理增强、Bug Fix |
| XY Plot | PR #3、#4、#5、#6 | XY 参数绘图套件、LoRA 输入、控件修复 |
| LyCORIS | PR #7、#8 | LOCON/LOHA/LOKR 训练方法、多方法配置 |
| 性能优化 | PR #9 | 训练循环与模块性能提升 |

### 代码量统计

| 指标 | 数值 |
|------|------|
| 新增文件 | ~20 个 |
| 修改文件 | ~25 个 |
| 新增代码行 | ~3500+ |
| 删除代码行 | ~200 |
| 涉及 PR | 9 个 |
| 活跃天数 | 8 天（05-17 ~ 05-24） |
