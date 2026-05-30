# Vue 3 技术冲突分析 + 执行清单

## 一、技术冲突分析

### 冲突 1：Vue 3 模板无法访问 `window.t` ❌ 严重

**问题：** Vue 3 的 runtime template 编译器生成的渲染函数，在 `with(this) { ... }` 作用域内求值。`this` 是组件实例的代理，只包含：
- 组件的 `data`/`props`/`computed`/`methods`
- `app.config.globalProperties` 上注册的属性
- `setup()` 返回的绑定

**`window.t` 不在其中。** 模板中写 `{{ t('app.title') }}` 会报 `t is not defined`。

**验证：** 现有代码中，`AnimaAPI`（全局变量）只在组件 `methods` 中使用，从未在 `template` 字符串中直接引用。这证实了 Vue 3 的 template 确实不能直接访问 `window` 全局变量。

**修复方案：** 在 `app.js` 中注册全局属性：
```javascript
app.config.globalProperties.t = window.t;
```
这样所有组件的模板都能通过 `this.t` → `t('key')` 访问翻译函数。

---

### 冲突 2：普通 JS 对象 `_messages` 不触发 Vue 响应式更新 ❌ 严重

**问题：** 当前方案中 `_messages` 是普通 JS 对象。即使 `t()` 在渲染期间被调用并读取了 `_messages.app.title`，Vue 3 的响应式系统无法追踪普通对象的属性访问。切换语言后，**组件不会自动重新渲染**。

**修复方案：** 使用 `Vue.reactive({})` 替代普通对象。`index.js` 在 Vue 之后加载（加载顺序：`vue.global.prod.js` → `i18n/index.js`），因此可以使用 `Vue.reactive`：

```javascript
var _messages = Vue.reactive({});
```

当 `t()` 在模板渲染期间被调用时，它会访问 `_messages` 的属性。由于 `_messages` 是 reactive proxy，Vue 的 effect 系统会追踪这些访问。切换语言时更新 `_messages`：

```javascript
function _setMessages(data) {
    Object.keys(_messages).forEach(function(k) { delete _messages[k]; });
    Object.assign(_messages, data);
}
```

Vue 3 会自动批量处理同步的 reactive 变更，只触发一次重新渲染。

---

### 冲突 3：`$forceUpdate` 不传播到子组件 ❌ 中等

**问题：** 计划中使用 `vm.$forceUpdate()` 触发重渲染。但 Vue 3 的 `$forceUpdate()` 只影响调用它的组件实例，**不会传播到子组件**。StageCard、LogViewer 等子组件中的 `t()` 调用不会更新。

**修复方案：** 采用冲突 2 的 reactive `_messages` 方案后，**完全不需要 `$forceUpdate`**。Vue 会自动追踪并更新所有访问了 `_messages` 的组件（包括子组件）。

---

### 冲突 4：模板字符串中的引号转义 ✅ 无冲突

**分析：** 现有模板使用单引号字符串数组：
```javascript
'  <span v-if="labelMarker === \'required\'" title="必填">*</span>',
```
改为 `:title="t('key')"` 后：
```javascript
'  <span v-if="labelMarker === \'required\'" :title="t(\'fieldRenderer.required\')">*</span>',
```
引号转义层级正确，无语法冲突。

---

### 冲突 5：`app.config.globalProperties` 在 Vue 3 prod build 中可用 ✅ 无冲突

**验证：** `vue.global.prod.js` 中包含 `globalProperties`（3 处引用）和 `Reactive`（9 处引用），确认 API 可用。

---

## 二、修正后的技术方案总结

| 项目 | 原计划 | 修正后 |
|------|--------|--------|
| `_messages` 类型 | 普通 JS 对象 | `Vue.reactive({})` |
| `t()` 模板可用性 | `window.t` 直接使用 | `app.config.globalProperties.t` |
| 语言切换重渲染 | `$forceUpdate` | Vue 响应式自动追踪 |
| `I18n.onChange` | 触发 `$forceUpdate` | **不再需要**（Vue 自动处理） |
| `I18n.init()` 回调 | `vm.$forceUpdate` | 仅设置 `document.documentElement.lang` |

---

## 三、执行清单

### Phase 0: 基础设施 (3 个文件)

#### Checklist 0.1: `workflow/i18n/locales/en.json`
- [ ] 创建文件，包含完整的英文翻译（~290 keys）
- [ ] 验证 JSON 语法合法
- [ ] 验证 key 层级结构：`app.*`, `fieldRenderer.*`, `schemaForm.*`, `methodSelector.*`, `stageCard.*`, `stageList.*`, `datasetSelector.*`, `runControl.*`, `infraSettings.*`, `logViewer.*`, `lossChart.*`, `langSwitcher.*`, `schema.*`, `backend.*`
- [ ] 提交

#### Checklist 0.2: `workflow/i18n/index.js` (修正版)
- [ ] 使用 `Vue.reactive({})` 替代普通 `_messages` 对象
- [ ] `_fallback` 保持普通对象（不变）
- [ ] 暴露 `window.t`（全局函数，供 methods 中直接调用）
- [ ] 暴露 `window.I18n`（含 init/setLocale/getLocale）
- [ ] `setLocale()` 内部调用 `_setMessages()` 更新 reactive 对象
- [ ] `init()` 返回 Promise，加载 fallback + current locale
- [ ] 验证：在浏览器 console 中 `window.t('app.welcomeTitle')` 返回字符串
- [ ] 提交

#### Checklist 0.3: `workflow/i18n/__init__.py` + `backend.py` + `schema_overlay.py`
- [ ] `backend.py`: `contextvars.ContextVar` 存 locale
- [ ] `backend.py`: `t(key, **params)` 函数，fallback en.json
- [ ] `backend.py`: `_load_locale()` 懒加载 JSON
- [ ] `__init__.py`: re-export `t`, `get_locale`, `set_locale`
- [ ] `schema_overlay.py`: `translate_schema(schema, schema_name, locale)`
- [ ] `schema_overlay.py`: 正确遍历 groups → fields → label/description/help/choice_labels/combo_switches
- [ ] 提交

---

### Phase 1: 前端集成 (2 个文件)

#### Checklist 1.1: `workflow/web/index.html`
- [ ] 在 `vue.global.prod.js` 之后、`api.js` 之前插入 `<script src="/static/i18n/index.js?v=5"></script>`
- [ ] 所有 script 标签的 `?v=4` 更新为 `?v=5`
- [ ] `<html lang="zh-CN">` 改为 `<html lang="en">`（动态更新）
- [ ] 提交

#### Checklist 1.2: `workflow/web/js/app.js` (启动逻辑)
- [ ] 在 `createApp()` 之后、`app.component()` 之前添加：
  ```javascript
  app.config.globalProperties.t = window.t;
  ```
- [ ] 替换末尾的 `var vm = app.mount("#app");` 为：
  ```javascript
  I18n.init().then(function() {
    document.documentElement.lang = I18n.getLocale();
    var vm = app.mount("#app");
  });
  ```
- [ ] 提交

---

### Phase 2: 前端字符串迁移 (11 个文件)

#### Checklist 2.1: `app.js` 模板中的字符串
逐条对照，确保每个中文字符串都有对应的 `t()` 替换：

- [ ] 行 673: `"Anima Workflow"` → `{{ t('app.welcomeTitle') }}`
- [ ] 行 677: `title="全局设置"` → `:title="t('app.settings')"`
- [ ] 行 680: `打开工作流 ▾` → `{{ t('app.openWorkflow') }}`
- [ ] 行 689: `暂无工作流` → `{{ t('app.noWorkflows') }}`
- [ ] 行 693: `新建工作流` → `{{ t('app.newWorkflow') }}`
- [ ] 行 694: `💾 保存` → `{{ t('app.save') }}`
- [ ] 行 701: `Anima Workflow` → `{{ t('app.welcomeTitle') }}`
- [ ] 行 703-704: 中文欢迎语 → `{{ t('app.welcomeDesc1') }}` + `{{ t('app.welcomeDesc2') }}`
- [ ] 行 707: `新建工作流` → `{{ t('app.newWorkflow') }}`
- [ ] 行 708: `打开工作流` → 文本替换
- [ ] 行 735: `💾 保存配置` → `{{ t('app.saveConfig') }}`
- [ ] 行 752: `选择左侧的阶段来编辑配置` → `{{ t('app.selectStage') }}`
- [ ] 行 760: `日志` → `{{ t('app.log') }}`
- [ ] 行 762: `系统日志` → `{{ t('app.systemLog') }}`
- [ ] 行 763: `脚本输出` → `{{ t('app.scriptOutput') }}`
- [ ] 行 764: `运行历史` → `{{ t('app.runHistory') }}`
- [ ] 行 766: `阶段` → 使用 `t('app.stages')`
- [ ] 行 772: `阶段 XX%` → `{{ t('app.stageProgress', {pct: overallProgress}) }}`
- [ ] 行 789: `等待运行...` → `{{ t('app.waitingToRun') }}`
- [ ] 行 794: `全部阶段` → `{{ t('app.allStages') }}`
- [ ] 行 797: `行` → 使用 `t('app.lines')`
- [ ] 行 804: `暂无脚本输出` → `{{ t('app.noScriptOutput') }}`
- [ ] 行 808: `暂无运行记录` → `{{ t('app.noRunHistory') }}`
- [ ] 行 812: 状态标签 (`✅ 完成` 等) → 使用 `t()` 表达式
- [ ] 行 824: `📋 日志` → 使用 `t('app.viewLog')`
- [ ] 行 839: `加载中...` → `{{ t('app.loading') }}`
- [ ] 行 842: `日志为空` → `{{ t('app.logEmpty') }}`
- [ ] 行 851: `⚙ 全局设置` → `{{ t('app.settingsTitle') }}`
- [ ] 行 855: `加载中...` → `{{ t('app.loading') }}`
- [ ] 行 859: `📁 工作流根目录` → `{{ t('app.workflowsRoot') }}`
- [ ] 行 863: `工作流根目录` → `{{ t('app.workflowsRootLabel') }}`
- [ ] 行 864: `留空使用默认路径` → `:placeholder="t('app.workflowsRootPlaceholder')"`
- [ ] 行 870: `🗂 模型路径` → `{{ t('app.modelPaths') }}`
- [ ] 行 874: `DiT 模型` → `{{ t('app.ditModel') }}`
- [ ] 行 878: `文本编码器 (qwen3)` → `{{ t('app.textEncoder') }}`
- [ ] 行 882: `VAE 模型` → `{{ t('app.vaeModel') }}`
- [ ] 行 889: `🔧 硬件设置` → `{{ t('app.hardwareSettings') }}`
- [ ] 行 893: `混合精度` → `{{ t('app.mixedPrecision') }}`
- [ ] 行 902: `注意力模式` → `{{ t('app.attnMode') }}`
- [ ] 行 913: `取消` → `{{ t('app.cancel') }}`
- [ ] 行 915: `保存中...` / `💾 保存设置` → 三元表达式 + `t()`
- [ ] 行 928: `取消` → `{{ t('app.cancel') }}`
- [ ] 行 929: `确定` → `{{ t('app.ok') }}`
- [ ] 提交

#### Checklist 2.2: `app.js` JS 逻辑中的字符串
- [ ] 行 102: `"打开失败: "` → `t("app.openFailed", {error: err})`
- [ ] 行 108: `"运行日志 — "` → `t("app.runLogTitle", {id: runId})`
- [ ] 行 116: `"加载日志失败: "` → `t("app.loadLogFailed", {error: ...})`
- [ ] 行 256: `"加载失败: "` → `t("app.loadFailed", {error: ...})`
- [ ] 行 286: `"工作流已保存"` → `t("app.workflowSaved")`
- [ ] 行 289: `"保存失败: "` → `t("app.saveFailed", {error: ...})`
- [ ] 行 297: `"新建工作流"` + `"输入工作流名称"` → `t()` 调用
- [ ] 行 306: `"工作流已创建: "` → `t("app.workflowCreated", {name: nm})`
- [ ] 行 311: `"创建失败: "` → `t("app.createFailed", {error: ...})`
- [ ] 行 386: `"运行失败: "` → `t("app.runFailed", {error: ...})`
- [ ] 行 394: `"正在停止..."` → `t("app.stopping")`
- [ ] 行 397: `"停止失败: "` → `t("app.stopFailed", {error: ...})`
- [ ] 行 416: `"▶ 工作流开始..."` → `t("app.workflowStart", {n: ...})`
- [ ] 行 420: `"▶ 阶段开始: "` → `t("app.stageStart", {id: ..., type: ...})`
- [ ] 行 465: `"💾 Checkpoint: "` → `t("app.checkpoint", {path: ..., epoch: ...})`
- [ ] 行 471: `"✅ 阶段完成: "` → `t("app.stageDone", {id: ...})`
- [ ] 行 474: `"❌ 阶段失败: "` → `t("app.stageFailed", {id: ..., status: ...})`
- [ ] 行 480: `"✅ 工作流完成"` → `t("app.workflowDone")`
- [ ] 行 481: `"工作流运行完成"` → `t("app.workflowRunDone")`
- [ ] 行 483: `"❌ 工作流失败"` → `t("app.workflowFail")`
- [ ] 行 484: `"工作流运行失败"` → `t("app.workflowRunFail")`
- [ ] 行 491: `"⚠ 连接断开"` → `t("app.connectionLost")`
- [ ] 行 568: `"设置已保存"` → `t("app.settingsSaved")`
- [ ] 行 571: `"保存失败: "` → `t("app.saveFailed", {error: ...})`
- [ ] 提交

#### Checklist 2.3: 语言切换器 (app.js)
- [ ] 添加 `var showLangMenu = ref(false);`
- [ ] 添加 `var currentLang = ref("en");`
- [ ] 添加 computed `currentLangLabel`
- [ ] 添加 `toggleLangMenu` / `switchLang` methods
- [ ] 在 header-right 模板中插入语言切换器 HTML
- [ ] 添加到 `return {}` 导出
- [ ] 提交

#### Checklist 2.4: 组件文件逐文件迁移

**FieldRenderer.js** (10 个字符串):
- [ ] `title="必填"` → `:title="t('fieldRenderer.required')"`
- [ ] `title="条件必填"` → `:title="t('fieldRenderer.conditionalRequired')"`
- [ ] `title="自动设置"` → `:title="t('fieldRenderer.autoSet')"`
- [ ] `'路径'` → `t('fieldRenderer.path')`
- [ ] `无上游预处理阶段` → `{{ t('fieldRenderer.noUpstreamPreprocess') }}`
- [ ] `不使用上游 checkpoint` → `{{ t('fieldRenderer.noUpstreamCheckpoint') }}`
- [ ] `placeholder="无上游训练阶段"` → `:placeholder="t('fieldRenderer.noUpstreamTrain')"`
- [ ] `"分析中..."` / `"分析数据集"` → `t()` 调用
- [ ] `"原始: "` + `"张"` / `"缩放后: "` + `"张"` → `t('fieldRenderer.original', {n: ...})` / `t('fieldRenderer.resized', {n: ...})`
- [ ] 提交

**SchemaForm.js** (1 个字符串):
- [ ] `加载 Schema...` → `{{ t('schemaForm.loadingSchema') }}`
- [ ] 提交

**MethodSelector.js** (1 个字符串):
- [ ] `训练方法` → `{{ t('methodSelector.trainingMethod') }}`
- [ ] 提交

**StageCard.js** (3 个字符串):
- [ ] `title="拖拽排序"` → `:title="t('stageCard.dragToReorder')"`
- [ ] `title="依赖"` → `:title="t('stageCard.dependencies')"`
- [ ] `title="删除阶段"` → `:title="t('stageCard.deleteStage')"`
- [ ] 提交

**StageList.js** (5 个字符串):
- [ ] `阶段面板` → `{{ t('stageList.stagePanel') }}`
- [ ] `暂无阶段，点击下方添加` → `{{ t('stageList.noStages') }}`
- [ ] `+ 添加阶段 ▾` → `{{ t('stageList.addStage') }}`
- [ ] `▶ 运行` → `{{ t('stageList.run') }}`
- [ ] `■ 停止` → `{{ t('stageList.stop') }}`
- [ ] 提交

**DatasetSelector.js** (3 个字符串):
- [ ] `暂无可用数据集` → `{{ t('datasetSelector.noDatasets') }}`
- [ ] `子集` → `{{ t('datasetSelector.subsets') }}`
- [ ] `重复:` → `{{ t('datasetSelector.repeat') }}`
- [ ] 提交

**RunControl.js** (3 个字符串):
- [ ] `▶ 运行` → `{{ t('runControl.run') }}`
- [ ] `■ 停止` → `{{ t('runControl.stop') }}`
- [ ] `阶段` → 使用 `t('runControl.stagesLabel')`
- [ ] 提交

**InfraSettings.js** (7 个字符串):
- [ ] `"加载基础设施配置失败: "` → `t('infraSettings.loadFailed', {error: ...})`
- [ ] `"保存失败: "` → `t('infraSettings.saveFailed', {error: ...})`
- [ ] `加载中...` → `{{ t('infraSettings.loading') }}`
- [ ] `📁 模型路径` → `{{ t('infraSettings.modelPaths') }}`
- [ ] `⚙ 硬件设置` → `{{ t('infraSettings.hardwareSettings') }}`
- [ ] `"保存中..."` / `"💾 保存基础设施设置"` → `t()` 三元表达式
- [ ] 提交

**LogViewer.js** (5 个字符串):
- [ ] `搜索日志...` → `:placeholder="t('logViewer.searchPlaceholder')"`
- [ ] `全部阶段` → `{{ t('logViewer.allStages') }}`
- [ ] `▼ 继续` → `{{ t('logViewer.resume') }}`
- [ ] `已暂停` → `{{ t('logViewer.paused') }}`
- [ ] `暂无日志` → `{{ t('logViewer.noLogs') }}`
- [ ] 提交

**LossChart.js** (4 个字符串):
- [ ] `Step: ` → `t('lossChart.step') + ': '`
- [ ] `Loss: ` → `t('lossChart.loss') + ': '`
- [ ] `LR: ` → `t('lossChart.lr') + ': '`
- [ ] `暂无训练数据` → `{{ t('lossChart.noData') }}`
- [ ] 提交

---

### Phase 3: 后端集成 (4 个文件)

#### Checklist 3.1: `workflow/app.py`
- [ ] 添加 `@web.middleware` 装饰的 `_locale_middleware`：解析 `Accept-Language` 头
- [ ] `create_app()` 中传入 `middlewares=[_locale_middleware]`
- [ ] `_handle_get_schema()` 中调用 `translate_schema(schema, schema_name, get_locale())`
- [ ] 提交

#### Checklist 3.2: `workflow/config.py`
- [ ] 添加 `from workflow.i18n import t`
- [ ] 5 个异常消息替换为 `t()` 调用
- [ ] 提交

#### Checklist 3.3: `workflow/scheduler.py`
- [ ] 添加 `from workflow.i18n import t`
- [ ] 3 个消息替换为 `t()` 调用
- [ ] 提交

#### Checklist 3.4: `workflow/models.py`
- [ ] 添加 `from workflow.i18n import t`
- [ ] 3 个验证消息替换为 `t()` 调用
- [ ] 提交

---

### Phase 4: 中文 + 日文翻译 (2 个文件)

#### Checklist 4.1: `workflow/i18n/locales/zh-CN.json`
- [ ] 结构与 `en.json` 完全一致（相同 key 层级）
- [ ] 每个值都是中文翻译（保留现有硬编码中文内容）
- [ ] **翻译对齐验证**：逐 key 对照 `en.json` 确保无遗漏
- [ ] 提交

#### Checklist 4.2: `workflow/i18n/locales/ja.json`
- [ ] 结构与 `en.json` 完全一致
- [ ] 每个值都是日文翻译
- [ ] **翻译对齐验证**：逐 key 对照 `en.json` 确保无遗漏
- [ ] 提交

---

### Phase 5: CSS + 收尾 (2 个文件)

#### Checklist 5.1: `workflow/web/css/style.css`
- [ ] 添加 `.lang-switcher` / `.lang-dropdown` 样式
- [ ] 提交

#### Checklist 5.2: 动态 `lang` 属性
- [ ] `index.html`: `<html lang="en">`
- [ ] `app.js` `I18n.init()` 回调中: `document.documentElement.lang = I18n.getLocale()`
- [ ] 提交

---

### Phase 6: 验证

#### Checklist 6.1: 浏览器测试
- [ ] 启动 workflow server
- [ ] 默认语言检测正确
- [ ] 切换 English → 所有 UI 文本变为英文
- [ ] 切换 日本語 → 所有 UI 文本变为日文
- [ ] 切回 中文 → 所有 UI 文本变回中文
- [ ] Schema 表单标签随语言切换
- [ ] 设置弹窗标签随语言切换
- [ ] 日志查看器占位符随语言切换
- [ ] 刷新页面后语言保持（localStorage）
- [ ] Fallback 验证：删除 `ja.json` 某个 key 后，日文界面显示英文 fallback

---

## 四、翻译内容对齐矩阵

### 前端翻译 key → 三语对照表（核心条目）

| Key | en | zh-CN | ja |
|-----|----|-------|----|
| `app.welcomeTitle` | Anima Workflow | Anima Workflow | Anima Workflow |
| `app.newWorkflow` | New Workflow | 新建工作流 | 新規ワークフロー |
| `app.openWorkflow` | Open Workflow ▾ | 打开工作流 ▾ | ワークフローを開く ▾ |
| `app.save` | 💾 Save | 💾 保存 | 💾 保存 |
| `app.workflowSaved` | Workflow saved | 工作流已保存 | ワークフローを保存しました |
| `app.saveFailed` | Save failed: {error} | 保存失败: {error} | 保存に失敗しました: {error} |
| `app.run` | ▶ Run | ▶ 运行 | ▶ 実行 |
| `app.stop` | ■ Stop | ■ 停止 | ■ 停止 |
| `app.stopping` | Stopping... | 正在停止... | 停止中... |
| `app.log` | Logs | 日志 | ログ |
| `app.systemLog` | System Log | 系统日志 | システムログ |
| `app.scriptOutput` | Script Output | 脚本输出 | スクリプト出力 |
| `app.runHistory` | Run History | 运行历史 | 実行履歴 |
| `app.settings` | ⚙ Settings | ⚙ 设置 | ⚙ 設定 |
| `app.settingsTitle` | ⚙ Global Settings | ⚙ 全局设置 | ⚙ グローバル設定 |
| `app.loading` | Loading... | 加载中... | 読み込み中... |
| `app.cancel` | Cancel | 取消 | キャンセル |
| `app.ok` | OK | 确定 | OK |
| `stageList.stagePanel` | Stages | 阶段面板 | ステージ |
| `stageList.addStage` | + Add Stage ▾ | + 添加阶段 ▾ | + ステージ追加 ▾ |
| `methodSelector.trainingMethod` | Training Method | 训练方法 | 学習方法 |
| `fieldRenderer.required` | Required | 必填 | 必須 |
| `logViewer.noLogs` | No logs | 暂无日志 | ログなし |
| `lossChart.noData` | No training data | 暂无训练数据 | 学習データなし |
| `langSwitcher.zh` | 中文 | 中文 | 中文 |
| `langSwitcher.en` | English | English | English |
| `langSwitcher.ja` | 日本語 | 日本語 | 日本語 |

### Schema 翻译 key → 三语对照表（核心条目）

| Key | en | zh-CN | ja |
|-----|----|-------|----|
| `schema.train_common.root.label` | Common Training Parameters | 训练通用参数 | 共通学習パラメータ |
| `schema.train_common.group.training` | Training Hyperparameters | 训练超参 | 学習ハイパーパラメータ |
| `schema.train_common.field.learning_rate` | Learning Rate | 学习率 | 学習率 |
| `schema.train_common.field.max_train_epochs` | Max Epochs | 最大 Epoch 数 | 最大エポック数 |
| `schema.preprocess.root.label` | Preprocessing | 预处理 | 前処理 |
| `schema.preprocess.field.source_image_dir` | Raw Dataset Path | 原始数据集路径 | 元データセットパス |
| `schema.infrastructure.root.label` | Infrastructure Configuration | 基础设施配置 | インフラ設定 |
| `schema.train_lora.root.label` | LoRA | LoRA | LoRA |
| `schema.train_lora.combo_switch.use_ortho.label` | OrthoLoRA | OrthoLoRA | OrthoLoRA |

> 注：技术术语（LoRA、OrthoLoRA、Network Dim 等）在三种语言中保持英文原样。
