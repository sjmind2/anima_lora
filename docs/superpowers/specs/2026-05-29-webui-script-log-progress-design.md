# Tasks

* [ ] Task 1: 项目骨架 + Pydantic 数据模型

  * [ ] SubTask 1.1: 写 models.py 的测试

  * [ ] SubTask 1.2: 运行测试确认失败

  * [ ] SubTask 1.3: 实现 models.py (WorkflowStage, WorkflowDefinition, StageOutput, SubsetInfo, InfrastructureConfig)

  * [ ] SubTask 1.4: 运行测试确认通过

  * [ ] SubTask 1.5: 提交

* [ ] Task 2: 配置管理（YAML/TOML 读写 + 占位符替换）

  * [ ] SubTask 2.1: 写 config.py 的测试

  * [ ] SubTask 2.2: 运行测试确认失败

  * [ ] SubTask 2.3: 实现 config.py (load/save YAML, load/save TOML, resolve\_placeholders)

  * [ ] SubTask 2.4: 运行测试确认通过

  * [ ] SubTask 2.5: 提交

* [ ] Task 3: Schema 加载器 + Schema YAML 文件

  * [ ] SubTask 3.1: 写 schema 加载器的测试

  * [ ] SubTask 3.2: 运行测试确认失败

  * [ ] SubTask 3.3: 创建所有 Schema YAML 文件 (preprocess, train\_common, train\_lokr, train\_lora, infrastructure)

  * [ ] SubTask 3.4: 在 config.py 中添加 load\_schema 函数

  * [ ] SubTask 3.5: 运行测试确认通过

  * [ ] SubTask 3.6: 提交

* [ ] Task 4: 统一日志 + SSE 事件队列

  * [ ] SubTask 4.1: 写 logger 的测试

  * [ ] SubTask 4.2: 运行测试确认失败

  * [ ] SubTask 4.3: 实现 logger.py (EventQueue, WorkflowLogger)

  * [ ] SubTask 4.4: 运行测试确认通过

  * [ ] SubTask 4.5: 提交

* [ ] Task 5: 阶段执行器（Preprocess + Train）

  * [ ] SubTask 5.1: 写阶段执行器的测试

  * [ ] SubTask 5.2: 运行测试确认失败

  * [ ] SubTask 5.3: 实现 base.py, preprocess.py, train.py

  * [ ] SubTask 5.4: 运行测试确认通过

  * [ ] SubTask 5.5: 提交

* [ ] Task 6: 工作流调度器

  * [ ] SubTask 6.1: 写调度器的测试

  * [ ] SubTask 6.2: 运行测试确认失败

  * [ ] SubTask 6.3: 实现 scheduler.py (WorkflowScheduler, topological sort, placeholder resolve, subprocess exec)

  * [ ] SubTask 6.4: 运行测试确认通过

  * [ ] SubTask 6.5: 提交

* [ ] Task 7: aiohttp HTTP 服务 + REST API + SSE

  * [ ] SubTask 7.1: 写 API 的测试

  * [ ] SubTask 7.2: 运行测试确认失败

  * [ ] SubTask 7.3: 实现 app.py (create\_app, all routes, SSE, static serving)

  * [ ] SubTask 7.4: 运行测试确认通过

  * [ ] SubTask 7.5: 提交

* [ ] Task 8: 入口 + pywebview 双模式

  * [ ] SubTask 8.1: 实现 __main__.py (--no-gui / pywebview 双模式)

  * [ ] SubTask 8.2: 添加 pywebview 到 pyproject.toml

  * [ ] SubTask 8.3: 手动验证启动

  * [ ] SubTask 8.4: 提交

* [ ] Task 9: 前端 WebUI — 基础框架 + 暗色主题

  * [ ] SubTask 9.1: 创建 index.html (Vue 3 CDN SPA 入口)

  * [ ] SubTask 9.2: 创建 style.css (暗色主题 + 按钮颜色 + 字体 + 间距)

  * [ ] SubTask 9.3: 创建 api.js (HTTP API 封装 + SSE 连接)

  * [ ] SubTask 9.4: 创建 app.js (Vue 3 应用主入口 + 全局状态)

  * [ ] SubTask 9.5: 手动验证暗色主题渲染

  * [ ] SubTask 9.6: 提交

* [ ] Task 10: 前端组件 — StageList + StageCard + SchemaForm

  * [ ] SubTask 10.1: 实现 StageList.js (阶段面板 + 拖拽排序 + 添加按钮)

  * [ ] SubTask 10.2: 实现 StageCard.js (状态图标 + 进度条 + 依赖标注)

  * [ ] SubTask 10.3: 实现 SchemaForm.js (动态表单 + Basic/Advanced + commonParams)

  * [ ] SubTask 10.4: 实现 FieldRenderer.js (按 type 分发控件)

  * [ ] SubTask 10.5: 实现 MethodSelector.js (基础类型 + 组合开关)

  * [ ] SubTask 10.6: 手动验证动态表单

  * [ ] SubTask 10.7: 提交

* [ ] Task 11: 前端组件 — DatasetSelector + LogViewer + LossChart + RunControl

  * [ ] SubTask 11.1: 实现 DatasetSelector.js (复选框 + 子集展开 + num\_repeats 编辑)

  * [ ] SubTask 11.2: 实现 LogViewer.js (自动滚动 + 暂停 + 搜索 + 过滤 + 彩色标记)

  * [ ] SubTask 11.3: 实现 LossChart.js (SVG 双层曲线 + EMA + 多阶段对比 + 降采样)

  * [ ] SubTask 11.4: 实现 RunControl.js (运行/停止 + 总进度)

  * [ ] SubTask 11.5: 实现 InfraSettings.js (基础设施配置面板)

  * [ ] SubTask 11.6: 手动验证完整操作流程

  * [ ] SubTask 11.7: 提交

* [ ] Task 12: CLI 脚本 + 配置模板

  * [ ] SubTask 12.1: 实现 run\_workflow\.py (CLI 运行)

  * [ ] SubTask 12.2: 实现 create\_workflow\.py (CLI 创建)

  * [ ] SubTask 12.3: 创建默认模板 TOML

  * [ ] SubTask 12.4: 手动验证 CLI

  * [ ] SubTask 12.5: 提交

* [ ] Task 13: 端到端验证 — 四阶段工作流

  * [ ] SubTask 13.1: 清除旧运行目录

  * [ ] SubTask 13.2: 创建工作流配置文件 (workflow\.yaml + 4 个 TOML)

  * [ ] SubTask 13.3: 通过 CLI 运行工作流

  * [ ] SubTask 13.4: 检查运行结果 (目录结构 + safetensors + 日志无报错)

  * [ ] SubTask 13.5: 提交

# Task Dependencies

* Task 2 依赖 Task 1

* Task 3 依赖 Task 2

* Task 4 独立

* Task 5 依赖 Task 1 + Task 2

* Task 6 依赖 Task 1 + Task 2 + Task 4 + Task 5

* Task 7 依赖 Task 3 + Task 6

* Task 8 依赖 Task 7

* Task 9 依赖 Task 7

* Task 10 依赖 Task 9

* Task 11 依赖 Task 10

* Task 12 依赖 Task 6

* Task 13 依赖 Task 1-12

