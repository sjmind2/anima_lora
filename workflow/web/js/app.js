(function () {
  var Vue = window.Vue;
  var createApp = Vue.createApp;
  var ref = Vue.ref;
  var reactive = Vue.reactive;
  var computed = Vue.computed;
  var watch = Vue.watch;
  var onMounted = Vue.onMounted;
  var nextTick = Vue.nextTick;

  var app = createApp({
    setup: function () {
      var workflowName = ref("");
      var workflowData = reactive({
        name: "",
        description: "",
        stages: [],
        infrastructure: {},
      });
      var selectedStageId = ref(null);
      var stageConfigs = reactive({});
      var runState = reactive({});
      var isRunning = ref(false);
      var eventSource = ref(null);
      var logLines = ref([]);
      var toasts = ref([]);
      var showWorkflowMenu = ref(false);
      var recentWorkflows = ref([]);
      var currentMethod = ref("train_lora");
      var methodCombos = reactive({});
      var activeTab = ref("config");
      var totalStages = ref(0);
      var completedStages = ref(0);
      var overallProgress = computed(function () {
        if (totalStages.value === 0) return 0;
        return Math.round((completedStages.value / totalStages.value) * 100);
      });
      var progressStatus = computed(function () {
        if (!isRunning.value && overallProgress.value === 100) return "done";
        if (!isRunning.value && completedStages.value > 0 && completedStages.value < totalStages.value) return "failed";
        if (isRunning.value) return "running";
        return "";
      });

      var selectedStage = computed(function () {
        if (!selectedStageId.value) return null;
        return workflowData.stages.find(function (s) {
          return s.id === selectedStageId.value;
        }) || null;
      });

      var selectedStageConfig = computed(function () {
        if (!selectedStageId.value) return {};
        return stageConfigs[selectedStageId.value] || {};
      });

      function getStageSchemaNames(stage) {
        if (!stage) return [];
        if (stage.type === "preprocess") return ["preprocess"];
        return [currentMethod.value, "train_common"];
      }

      function showToast(msg, type) {
        var t = { msg: msg, type: type || "info", id: Date.now() };
        toasts.value.push(t);
        setTimeout(function () {
          var idx = toasts.value.indexOf(t);
          if (idx >= 0) toasts.value.splice(idx, 1);
        }, 3000);
      }

      function generateId(type) {
        return type + "_" + Date.now().toString(36);
      }

      function loadRecentWorkflows() {
        AnimaAPI.getRecentWorkflows()
          .then(function (list) {
            recentWorkflows.value = list;
          })
          .catch(function () {});
      }

      function loadWorkflow(name) {
        workflowName.value = name;
        AnimaAPI.getWorkflow(name)
          .then(function (data) {
            Object.assign(workflowData, data);
            selectedStageId.value = null;
            logLines.value = [];
            resetRunState();
            loadStageConfigs();
          })
          .catch(function (err) {
            showToast("加载失败: " + (err.error || err), "error");
          });
      }

      function loadStageConfigs() {
        workflowData.stages.forEach(function (stage) {
          if (!stageConfigs[stage.id]) {
            stageConfigs[stage.id] = {};
          }
        });
      }

      function saveWorkflow() {
        if (!workflowName.value) return;
        var data = {
          name: workflowData.name || workflowName.value,
          description: workflowData.description || "",
          stages: JSON.parse(JSON.stringify(workflowData.stages)),
          infrastructure: Object.assign({}, workflowData.infrastructure),
        };
        AnimaAPI.updateWorkflow(workflowName.value, data)
          .then(function () {
            showToast("工作流已保存", "success");
          })
          .catch(function (err) {
            showToast("保存失败: " + (err.error || err), "error");
          });
      }

      function createNewWorkflow(name) {
        var nm = name || prompt("输入工作流名称:");
        if (!nm) return;
        nm = nm.trim();
        if (!nm) return;
        AnimaAPI.createWorkflow(nm)
          .then(function (data) {
            showToast("工作流已创建: " + nm, "success");
            loadWorkflow(nm);
            loadRecentWorkflows();
          })
          .catch(function (err) {
            showToast("创建失败: " + (err.error || err), "error");
          });
      }

      function addStage(type) {
        var id = generateId(type);
        var configFileName = id + ".toml";
        var stage = {
          id: id,
          type: type,
          config_file: configFileName,
          depends_on: [],
        };
        workflowData.stages.push(stage);
        stageConfigs[stage.id] = {};
        selectedStageId.value = stage.id;
        saveWorkflow();
      }

      function removeStage(stageId) {
        var idx = workflowData.stages.findIndex(function (s) {
          return s.id === stageId;
        });
        if (idx < 0) return;
        workflowData.stages.splice(idx, 1);
        delete stageConfigs[stageId];
        if (selectedStageId.value === stageId) {
          selectedStageId.value = workflowData.stages.length > 0
            ? workflowData.stages[0].id
            : null;
        }
        saveWorkflow();
      }

      function selectStage(stageId) {
        selectedStageId.value = stageId;
      }

      function reorderStages(fromIdx, toIdx) {
        var stage = workflowData.stages.splice(fromIdx, 1)[0];
        workflowData.stages.splice(toIdx, 0, stage);
        saveWorkflow();
      }

      function resetRunState() {
        Object.keys(runState).forEach(function (k) {
          delete runState[k];
        });
        totalStages.value = 0;
        completedStages.value = 0;
        isRunning.value = false;
      }

      function startRun() {
        if (!workflowName.value) return;
        resetRunState();
        isRunning.value = true;
        logLines.value = [];

        AnimaAPI.runWorkflow(workflowName.value)
          .then(function () {
            connectSSE();
          })
          .catch(function (err) {
            isRunning.value = false;
            showToast("运行失败: " + (err.error || err), "error");
          });
      }

      function stopRun() {
        if (!workflowName.value) return;
        AnimaAPI.stopWorkflow(workflowName.value)
          .then(function () {
            showToast("正在停止...", "info");
          })
          .catch(function (err) {
            showToast("停止失败: " + (err.error || err), "error");
          });
      }

      function connectSSE() {
        if (eventSource.value) {
          eventSource.value.close();
        }
        var runId = "latest";
        eventSource.value = AnimaAPI.connectEventStream(runId, function (ev) {
          handleEvent(ev);
        });
      }

      function handleEvent(ev) {
        switch (ev.ev) {
          case "workflow_start":
            totalStages.value = ev.total_stages || 0;
            completedStages.value = 0;
            addLog("▶ 工作流开始 (" + (ev.total_stages || 0) + " 个阶段)", "info");
            break;
          case "stage_start":
            runState[ev.stage_id] = { status: "running", progress: 0 };
            addLog("▶ 阶段开始: " + ev.stage_id + " (" + ev.stage_type + ")", "stage-start");
            break;
          case "stage_progress":
            if (runState[ev.stage_id]) {
              runState[ev.stage_id].progress = ev.progress || 0;
            }
            break;
          case "stage_ckpt":
            addLog("💾 Checkpoint: " + ev.path + " (epoch " + ev.epoch + ")", "info");
            break;
          case "stage_end":
            if (ev.status === "ok") {
              runState[ev.stage_id] = { status: "done", progress: 100 };
              completedStages.value++;
              addLog("✅ 阶段完成: " + ev.stage_id, "success");
            } else {
              runState[ev.stage_id] = { status: "failed", progress: 0 };
              addLog("❌ 阶段失败: " + ev.stage_id + " — " + ev.status, "error");
            }
            break;
          case "workflow_end":
            isRunning.value = false;
            if (ev.status === "ok") {
              addLog("✅ 工作流完成", "success");
              showToast("工作流运行完成", "success");
            } else {
              addLog("❌ 工作流失败", "error");
              showToast("工作流运行失败", "error");
            }
            if (eventSource.value) {
              eventSource.value.close();
              eventSource.value = null;
            }
            break;
          case "stream_error":
            isRunning.value = false;
            addLog("⚠ 连接断开", "error");
            break;
        }
      }

      function addLog(text, cls) {
        logLines.value.push({ text: text, cls: cls || "", ts: new Date().toLocaleTimeString() });
        nextTick(function () {
          var el = document.querySelector(".log-viewer");
          if (el) el.scrollTop = el.scrollHeight;
        });
      }

      function updateStageConfig(newVal) {
        if (selectedStageId.value) {
          stageConfigs[selectedStageId.value] = newVal;
        }
      }

      function onMethodChange(methodName) {
        currentMethod.value = methodName;
      }

      function toggleWorkflowMenu() {
        showWorkflowMenu.value = !showWorkflowMenu.value;
        if (showWorkflowMenu.value) {
          loadRecentWorkflows();
        }
      }

      function onStageConfigTabChange(tab) {
        activeTab.value = tab;
      }

      return {
        workflowName: workflowName,
        workflowData: workflowData,
        selectedStageId: selectedStageId,
        stageConfigs: stageConfigs,
        runState: runState,
        isRunning: isRunning,
        logLines: logLines,
        toasts: toasts,
        showWorkflowMenu: showWorkflowMenu,
        recentWorkflows: recentWorkflows,
        currentMethod: currentMethod,
        methodCombos: methodCombos,
        activeTab: activeTab,
        totalStages: totalStages,
        completedStages: completedStages,
        overallProgress: overallProgress,
        progressStatus: progressStatus,
        selectedStage: selectedStage,
        selectedStageConfig: selectedStageConfig,
        getStageSchemaNames: getStageSchemaNames,
        loadWorkflow: loadWorkflow,
        saveWorkflow: saveWorkflow,
        createNewWorkflow: createNewWorkflow,
        addStage: addStage,
        removeStage: removeStage,
        selectStage: selectStage,
        reorderStages: reorderStages,
        startRun: startRun,
        stopRun: stopRun,
        updateStageConfig: updateStageConfig,
        onMethodChange: onMethodChange,
        toggleWorkflowMenu: toggleWorkflowMenu,
        loadRecentWorkflows: loadRecentWorkflows,
        onStageConfigTabChange: onStageConfigTabChange,
      };
    },
    mounted: function() {
      var self = this;
      window.__wf = {
        createNewWorkflow: function(n) { self.createNewWorkflow(n); },
        loadWorkflow: function(n) { self.loadWorkflow(n); },
        addStage: function(t) { self.addStage(t); },
        removeStage: function(id) { self.removeStage(id); },
        selectStage: function(id) { self.selectStage(id); },
        saveWorkflow: function() { self.saveWorkflow(); },
        startRun: function() { self.startRun(); },
        stopRun: function() { self.stopRun(); },
      };
    },
    template: [
      '<div id="app-root" style="display:flex;flex-direction:column;height:100vh;">',

      '  <header class="header">',
      '    <div class="header-left">',
      '      <span class="header-title"><span class="icon">🔄</span> Anima Workflow</span>',
      '      <span v-if="workflowName" style="color:var(--text-dim);font-size:13px;">— {{ workflowName }}</span>',
      '    </div>',
      '    <div class="header-right">',
      '      <div class="dropdown">',
      '        <button class="btn btn-ghost btn-sm" @click="toggleWorkflowMenu">',
      '          打开工作流 ▾',
      '        </button>',
      '        <div v-if="showWorkflowMenu" class="dropdown-menu">',
      '          <button class="dropdown-item"',
      '            v-for="wf in recentWorkflows" :key="wf.dir || wf.name"',
      '            @click="loadWorkflow(wf.dir || wf.name); showWorkflowMenu = false;">',
      '            {{ wf.name || wf.dir }}',
      '          </button>',
      '          <div v-if="recentWorkflows.length === 0" style="padding:8px 14px;color:var(--text-dim);font-size:12px;">',
      '            暂无工作流',
      '          </div>',
      '        </div>',
      '      </div>',
      '      <button class="btn btn-blue btn-sm" @click="createNewWorkflow()">新建工作流</button>',
      '      <button v-if="workflowName" class="btn btn-ghost btn-sm" @click="saveWorkflow">💾 保存</button>',
      '    </div>',
      '  </header>',

      '  <div v-if="!workflowName" class="main-content">',
      '    <div class="welcome-screen">',
      '      <div class="welcome-icon">🔄</div>',
      '      <div class="welcome-title">Anima Workflow</div>',
      '      <div class="welcome-desc">',
      '        创建或打开一个工作流来开始管理你的 LoRA 训练流水线。',
      '        支持预处理、训练阶段的拖拽排序和依赖管理。',
      '      </div>',
      '      <div class="welcome-actions">',
      '        <button class="btn btn-blue" @click="createNewWorkflow()">新建工作流</button>',
      '        <button class="btn btn-ghost" @click="toggleWorkflowMenu(); loadRecentWorkflows();">打开工作流</button>',
      '      </div>',
      '    </div>',
      '  </div>',

      '  <div v-if="workflowName" class="main-content">',
      '    <div class="stage-panel">',
      '      <stage-list',
      '        :stages="workflowData.stages"',
      '        :selectedId="selectedStageId"',
      '        :runState="runState"',
      '        :isRunning="isRunning"',
      '        @select="selectStage"',
      '        @add="addStage"',
      '        @remove="removeStage"',
      '        @reorder="reorderStages"',
      '        @run="startRun"',
      '        @stop="stopRun" />',
      '    </div>',
      '    <div class="config-panel">',
      '      <div v-if="selectedStage" class="config-panel-content">',
      '        <div class="config-panel-header">',
      '          <span class="config-panel-title">',
      '            {{ selectedStage.type === "train" ? "🎯" : "📁" }}',
      '            {{ selectedStage.type === "train" ? "Train" : "Preprocess" }}',
      '            — {{ selectedStage.id }}',
      '          </span>',
      '          <div class="config-panel-actions">',
      '            <button class="btn btn-ghost btn-sm" @click="saveWorkflow">💾 保存配置</button>',
      '          </div>',
      '        </div>',
      '        <method-selector',
      '          v-if="selectedStage.type === \'train\'"',
      '          v-model="currentMethod"',
      '          v-model:combos="methodCombos"',
      '          @change="onMethodChange" />',
      '        <schema-form',
      '          :schemaNames="getStageSchemaNames(selectedStage)"',
      '          :modelValue="selectedStageConfig"',
      '          @update:modelValue="updateStageConfig" />',
      '      </div>',
      '      <div v-if="!selectedStage && workflowName" class="config-panel-content">',
      '        <div class="empty-state" style="margin-top:80px;">',
      '          选择左侧的阶段来编辑配置',
      '        </div>',
      '      </div>',
      '    </div>',
      '  </div>',

      '  <div v-if="workflowName" class="bottom-panel">',
      '    <div class="bottom-panel-header">',
      '      <span class="bottom-panel-title">日志</span>',
      '      <span style="font-size:11px;color:var(--text-dim);">{{ completedStages }}/{{ totalStages }} 阶段</span>',
      '    </div>',
      '    <div v-if="isRunning || completedStages > 0" class="progress-bar-container">',
      '      <div class="progress-bar-track">',
      '        <div class="progress-bar-fill" :class="progressStatus" :style="{ width: overallProgress + \'%\' }"></div>',
      '      </div>',
      '      <div class="progress-label">{{ overallProgress }}%</div>',
      '    </div>',
      '    <div class="log-viewer">',
      '      <div v-for="(line, i) in logLines" :key="i" class="log-line" :class="line.cls">',
      '        [{{ line.ts }}] {{ line.text }}',
      '      </div>',
      '      <div v-if="logLines.length === 0" style="color:var(--text-dim);font-style:italic;">等待运行...</div>',
      '    </div>',
      '  </div>',

      '  <div v-for="toast in toasts" :key="toast.id" class="toast" :class="\'toast-\' + toast.type">',
      '    {{ toast.msg }}',
      '  </div>',

      '</div>',
    ].join("\n"),
  });

  var COMPONENT_MAP = {
    FieldRenderer: "field-renderer",
    SchemaForm: "schema-form",
    MethodSelector: "method-selector",
    StageCard: "stage-card",
    StageList: "stage-list",
    DatasetSelector: "dataset-selector",
    LogViewer: "log-viewer",
    LossChart: "loss-chart",
    RunControl: "run-control",
    InfraSettings: "infra-settings",
  };

  var components = window.VueComponents || {};
  Object.keys(COMPONENT_MAP).forEach(function (className) {
    if (components[className]) {
      app.component(COMPONENT_MAP[className], components[className]);
    }
  });

  var vm = app.mount("#app");

  window.__workflowApp = {
    createNewWorkflow: function(name) { window.__wf && window.__wf.createNewWorkflow(name); },
    loadWorkflow: function(name) { window.__wf && window.__wf.loadWorkflow(name); },
    addStage: function(type) { window.__wf && window.__wf.addStage(type); },
    selectStage: function(id) { window.__wf && window.__wf.selectStage(id); },
    saveWorkflow: function() { window.__wf && window.__wf.saveWorkflow(); },
    startRun: function() { window.__wf && window.__wf.startRun(); },
    stopRun: function() { window.__wf && window.__wf.stopRun(); },
  };
})();
