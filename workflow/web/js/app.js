(function () {
  var Vue = window.Vue;
  var createApp = Vue.createApp;
  var ref = Vue.ref;
  var reactive = Vue.reactive;
  var computed = Vue.computed;
  var watch = Vue.watch;
  var onMounted = Vue.onMounted;
  var nextTick = Vue.nextTick;

  var currentLang = ref("en");
  var showLangMenu = ref(false);

  var currentLangLabel = computed(function () {
    var locale = currentLang.value;
    return t("langSwitcher." + locale, locale);
  });

  function toggleLangMenu() {
    showLangMenu.value = !showLangMenu.value;
  }

  function switchLang(locale) {
    I18n.setLocale(locale);
    currentLang.value = locale;
    showLangMenu.value = false;
    document.documentElement.lang = locale;
  }

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
      var scriptLogs = ref({});
      var activeScriptStage = ref(null);
      var showLogModal = ref(false);
      var logModalLines = ref([]);
      var logModalTitle = ref("");
      var logModalLoading = ref(false);
      var scriptProgress = reactive({
        pct: 0, current: 0, total: 0,
        elapsed: "", eta: "", rate: "",
        metrics: {}, rawLine: "", active: false,
      });
      var activeLogTab = ref("system");
      var _scriptId = 0;
      var runHistory = ref([]);
      var showModal = ref(false);
      var modalTitle = ref("");
      var modalInput = ref("");
      var modalPlaceholder = ref("");
      var _modalResolve = null;
      var showSettingsModal = ref(false);
      var settingsData = reactive({
        workflows_root: "",
        pretrained_model_name_or_path: "",
        qwen3: "",
        vae: "",
        mixed_precision: "bf16",
        attn_mode: "flex",
      });
      var settingsLoading = ref(false);
      var settingsSaving = ref(false);

      function modalPrompt(title, placeholder) {
        return new Promise(function(resolve) {
          modalTitle.value = title;
          modalPlaceholder.value = placeholder || "";
          modalInput.value = "";
          showModal.value = true;
          _modalResolve = resolve;
        });
      }

      function modalConfirm() {
        showModal.value = false;
        if (_modalResolve) {
          var val = modalInput.value.trim();
          _modalResolve(val || null);
          _modalResolve = null;
        }
      }

      function modalCancel() {
        showModal.value = false;
        if (_modalResolve) {
          _modalResolve(null);
          _modalResolve = null;
        }
      }

      function loadRunHistory() {
        if (!workflowName.value) return;
        AnimaAPI.listRuns(workflowName.value)
          .then(function(runs) { runHistory.value = runs.sort(function(a, b) { return b.id.localeCompare(a.id); }); })
          .catch(function() { runHistory.value = []; });
      }

      function openRunDir(runId) {
        if (!workflowName.value) return;
        AnimaAPI.openRunDir(workflowName.value, runId)
          .catch(function(err) { showToast(t("app.openFailed", {error: err}), "error"); });
      }

      function viewRunLog(runId) {
        if (!workflowName.value) return;
        showLogModal.value = true;
        logModalTitle.value = t("app.runLogTitle", {id: runId});
        logModalLines.value = [];
        logModalLoading.value = true;
        AnimaAPI.getWorkflowRunLog(workflowName.value, runId)
          .then(function(data) {
            logModalLines.value = data.lines || [];
          })
          .catch(function(err) {
            logModalLines.value = [t("app.loadLogFailed", {error: err.error || err})];
          })
          .finally(function() {
            logModalLoading.value = false;
          });
      }

      function closeLogModal() {
        showLogModal.value = false;
        logModalLines.value = [];
      }
      var overallProgress = computed(function () {
        if (totalStages.value === 0) return 0;
        return Math.round((completedStages.value / totalStages.value) * 100);
      });

      var flatScriptLogs = computed(function () {
        var all = [];
        var stageIds = Object.keys(scriptLogs.value).sort();
        stageIds.forEach(function(sid) {
          var lines = scriptLogs.value[sid] || [];
          lines.forEach(function(l) { all.push(l); });
        });
        return all;
      });

      var scriptStageIds = computed(function () {
        return Object.keys(scriptLogs.value).sort();
      });

      var filteredScriptLogs = computed(function () {
        var stage = activeScriptStage.value;
        if (!stage) return flatScriptLogs.value;
        return scriptLogs.value[stage] || [];
      });

      var flatScriptLogCount = computed(function () {
        var n = 0;
        Object.keys(scriptLogs.value).forEach(function(sid) {
          n += (scriptLogs.value[sid] || []).length;
        });
        return n;
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
        var existing = workflowData.stages.filter(function(s) { return s.type === type; }).length;
        return type + "_" + (existing + 1);
      }

      var TQDM_RE = /^(\S+):\s+(\d+)%\|[^|]*\|\s+(\d+)\/(\d+)\s+\[([^\]]+)\](?:\s+(.+))?/;
      var TQDM_METRIC_RE = /(\w[\w_]*)=([^\s,]+)/g;

      function parseTqdmLine(line) {
        var m = line.match(TQDM_RE);
        if (!m) return null;
        var result = {
          prefix: m[1], pct: parseInt(m[2]),
          current: parseInt(m[3]), total: parseInt(m[4]),
          timing: m[5], extra: m[6] || ""
        };
        var tm = result.timing.match(/^([^<]+)<([^,]+),\s*(.+)$/);
        if (tm) {
          result.elapsed = tm[1].trim();
          result.eta = tm[2].trim();
          result.rate = tm[3].trim();
        }
        result.metrics = {};
        TQDM_METRIC_RE.lastIndex = 0;
        var mm;
        while ((mm = TQDM_METRIC_RE.exec(result.extra)) !== null) {
          result.metrics[mm[1]] = mm[2];
        }
        return result;
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
            var loadedConfigs = data.stage_configs || {};
            delete data.stage_configs;
            Object.assign(workflowData, data);
            selectedStageId.value = null;
            logLines.value = [];
            resetRunState();
            workflowData.stages.forEach(function (stage, idx) {
              if (!stage.label) {
                var typeCount = workflowData.stages.slice(0, idx + 1).filter(function(s) { return s.type === stage.type; }).length;
                var typeLabel = stage.type === "train" ? "Train" : "Preprocess";
                stage.label = typeLabel + " " + typeCount;
              }
              if (loadedConfigs[stage.id]) {
                stageConfigs[stage.id] = loadedConfigs[stage.id];
              } else if (!stageConfigs[stage.id]) {
                stageConfigs[stage.id] = {};
              }
            });
          })
          .catch(function (err) {
            showToast(t("app.loadFailed", {error: err.error || err}), "error");
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
        var configsCopy = {};
        Object.keys(stageConfigs).forEach(function(k) {
          var v = stageConfigs[k];
          if (v && typeof v === "object") {
            configsCopy[k] = JSON.parse(JSON.stringify(v));
          }
        });
        var data = {
          name: workflowData.name || workflowName.value,
          description: workflowData.description || "",
          stages: JSON.parse(JSON.stringify(workflowData.stages)),
          infrastructure: Object.assign({}, workflowData.infrastructure),
          stage_configs: configsCopy,
        };
        AnimaAPI.updateWorkflow(workflowName.value, data)
          .then(function () {
            showToast(t("app.workflowSaved"), "success");
          })
          .catch(function (err) {
            showToast(t("app.saveFailed", {error: err.error || err}), "error");
          });
      }

      function createNewWorkflow(name) {
        if (name) {
          _doCreate(name);
        } else {
          modalPrompt(t("app.newWorkflow"), t("app.enterWorkflowName")).then(function(nm) {
            if (nm) _doCreate(nm);
          });
        }
      }

      function _doCreate(nm) {
        AnimaAPI.createWorkflow(nm)
          .then(function (data) {
            showToast(t("app.workflowCreated", {name: nm}), "success");
            loadWorkflow(nm);
            loadRecentWorkflows();
          })
          .catch(function (err) {
            showToast(t("app.createFailed", {error: err.error || err}), "error");
          });
      }

      function addStage(type) {
        var id = generateId(type);
        var configFileName = id + ".toml";
        var typeCount = workflowData.stages.filter(function(s) { return s.type === type; }).length + 1;
        var typeLabel = type === "train" ? "Train" : "Preprocess";
        var label = typeLabel + " " + typeCount;
        var stage = {
          id: id,
          type: type,
          config_file: configFileName,
          depends_on: [],
          label: label,
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
        var stage = workflowData.stages.find(function(s) { return s.id === stageId; });
        if (stage && stage.type === "train") {
          var cfg = stageConfigs[stageId] || {};
          var nt = cfg.network_type || "lora";
          currentMethod.value = "train_" + nt;
        }
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
            showToast(t("app.runFailed", {error: err.error || err}), "error");
          });
      }

      function stopRun() {
        if (!workflowName.value) return;
        AnimaAPI.stopWorkflow(workflowName.value)
          .then(function () {
            showToast(t("app.stopping"), "info");
          })
          .catch(function (err) {
            showToast(t("app.stopFailed", {error: err.error || err}), "error");
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
            addLog(t("app.workflowStart", {n: ev.total_stages || 0}), "info");
            break;
          case "stage_start":
            runState[ev.stage_id] = { status: "running", progress: 0 };
            addLog(t("app.stageStart", {id: ev.stage_id, type: ev.stage_type}), "stage-start");
            scriptProgress.active = false;
            scriptProgress.rawLine = "";
            scriptProgress.pct = 0;
            scriptLogs.value[ev.stage_id] = [];
            scriptLogs.value = Object.assign({}, scriptLogs.value);
            break;
          case "stage_progress":
            if (runState[ev.stage_id]) {
              runState[ev.stage_id].progress = ev.progress || 0;
            }
            break;
          case "stage_stdout_batch":
            ev.lines.forEach(function(line) {
              var parsed = parseTqdmLine(line);
              if (parsed) {
                Object.assign(scriptProgress, {
                  pct: parsed.pct,
                  current: parsed.current,
                  total: parsed.total,
                  elapsed: parsed.elapsed,
                  eta: parsed.eta,
                  rate: parsed.rate,
                  metrics: parsed.metrics,
                  rawLine: line,
                  active: true,
                });
              } else {
                if (!scriptLogs.value[ev.stage_id]) {
                  scriptLogs.value[ev.stage_id] = [];
                }
                scriptLogs.value[ev.stage_id].push({
                  _id: ++_scriptId,
                  text: line,
                  ts: new Date().toLocaleTimeString(),
                  stage_id: ev.stage_id,
                });
                if (scriptLogs.value[ev.stage_id].length > 500) {
                  scriptLogs.value[ev.stage_id] = scriptLogs.value[ev.stage_id].slice(-400);
                }
                scriptLogs.value = Object.assign({}, scriptLogs.value);
              }
            });
            break;
          case "stage_ckpt":
            addLog(t("app.checkpoint", {path: ev.path, epoch: ev.epoch}), "info");
            break;
          case "stage_end":
            if (ev.status === "ok") {
              runState[ev.stage_id] = { status: "done", progress: 100 };
              completedStages.value++;
              addLog(t("app.stageDone", {id: ev.stage_id}), "success");
            } else {
              runState[ev.stage_id] = { status: "failed", progress: 0 };
              addLog(t("app.stageFailed", {id: ev.stage_id, status: ev.status}), "error");
            }
            break;
          case "workflow_end":
            isRunning.value = false;
            if (ev.status === "ok") {
              addLog(t("app.workflowDone"), "success");
              showToast(t("app.workflowRunDone"), "success");
            } else {
              addLog(t("app.workflowFail"), "error");
              showToast(t("app.workflowRunFail"), "error");
            }
            if (eventSource.value) { eventSource.value.close(); eventSource.value = null; }
            loadRunHistory();
            break;
          case "stream_error":
            isRunning.value = false;
            addLog(t("app.connectionLost"), "error");
            break;
        }
      }

      function addLog(text, cls) {
        logLines.value.push({ text: text, cls: cls || "", ts: new Date().toLocaleTimeString() });
        if (activeLogTab.value === "system") {
          nextTick(function () {
            var el = document.querySelector(".log-viewer");
            if (el) el.scrollTop = el.scrollHeight;
          });
        }
      }

      function updateStageConfig(newVal) {
        if (selectedStageId.value) {
          var existing = stageConfigs[selectedStageId.value];
          if (existing) {
            Object.keys(newVal).forEach(function(k) { existing[k] = newVal[k]; });
          } else {
            stageConfigs[selectedStageId.value] = newVal;
          }
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

      function openSettings() {
        showSettingsModal.value = true;
        settingsLoading.value = true;
        var infraPromise = workflowName.value
          ? AnimaAPI.getInfra(workflowName.value).catch(function() { return {}; })
          : Promise.resolve({});
        var settingsPromise = AnimaAPI.getSettings().catch(function() { return {}; });
        Promise.all([infraPromise, settingsPromise]).then(function(results) {
          var infra = results[0] || {};
          var settings = results[1] || {};
          settingsData.workflows_root = settings.workflows_root || "";
          settingsData.pretrained_model_name_or_path = infra.pretrained_model_name_or_path || "";
          settingsData.qwen3 = infra.qwen3 || "";
          settingsData.vae = infra.vae || "";
          settingsData.mixed_precision = infra.mixed_precision || "bf16";
          settingsData.attn_mode = infra.attn_mode || "flex";
        }).finally(function() {
          settingsLoading.value = false;
        });
      }

      function saveSettings() {
        settingsSaving.value = true;
        var settingsPayload = { workflows_root: settingsData.workflows_root };
        var infraPayload = {
          pretrained_model_name_or_path: settingsData.pretrained_model_name_or_path,
          qwen3: settingsData.qwen3,
          vae: settingsData.vae,
          mixed_precision: settingsData.mixed_precision,
          attn_mode: settingsData.attn_mode,
        };
        var promises = [AnimaAPI.setSettings(settingsPayload)];
        if (workflowName.value) {
          promises.push(AnimaAPI.setInfra(workflowName.value, infraPayload));
        }
        Promise.all(promises).then(function() {
          showToast(t("app.settingsSaved"), "success");
          showSettingsModal.value = false;
        }).catch(function(err) {
          showToast(t("app.saveFailed", {error: err.error || err}), "error");
        }).finally(function() {
          settingsSaving.value = false;
        });
      }

      function closeSettings() {
        showSettingsModal.value = false;
      }

      watch(activeLogTab, function() {
        nextTick(function () {
          var el = document.querySelector(".log-viewer");
          if (el) el.scrollTop = el.scrollHeight;
        });
      });

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
        scriptLogs: scriptLogs,
        activeScriptStage: activeScriptStage,
        scriptStageIds: scriptStageIds,
        filteredScriptLogs: filteredScriptLogs,
        flatScriptLogs: flatScriptLogs,
        flatScriptLogCount: flatScriptLogCount,
        scriptProgress: scriptProgress,
        activeLogTab: activeLogTab,
        runHistory: runHistory,
        loadRunHistory: loadRunHistory,
        openRunDir: openRunDir,
        viewRunLog: viewRunLog,
        closeLogModal: closeLogModal,
        showLogModal: showLogModal,
        logModalLines: logModalLines,
        logModalTitle: logModalTitle,
        logModalLoading: logModalLoading,
        showModal: showModal,
        modalTitle: modalTitle,
        modalInput: modalInput,
        modalPlaceholder: modalPlaceholder,
        modalConfirm: modalConfirm,
        modalCancel: modalCancel,
        showSettingsModal: showSettingsModal,
        settingsData: settingsData,
        settingsLoading: settingsLoading,
        settingsSaving: settingsSaving,
        openSettings: openSettings,
        saveSettings: saveSettings,
        closeSettings: closeSettings,
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
        currentLang: currentLang,
        showLangMenu: showLangMenu,
        currentLangLabel: currentLangLabel,
        toggleLangMenu: toggleLangMenu,
        switchLang: switchLang,
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
      '      <div class="lang-switcher">',
      '        <button class="btn btn-ghost btn-sm" @click="toggleLangMenu">',
      '          🌐 {{ currentLangLabel }}',
      '        </button>',
      '        <div v-if="showLangMenu" class="lang-dropdown">',
      '          <button class="dropdown-item" @click="switchLang(\'zh-CN\')">中文</button>',
      '          <button class="dropdown-item" @click="switchLang(\'en\')">English</button>',
      '          <button class="dropdown-item" @click="switchLang(\'ja\')">日本語</button>',
      '        </div>',
      '      </div>',
      '      <button class="btn btn-ghost btn-sm" @click="openSettings" :title="t(\'app.settings\')">⚙</button>',
      '      <div class="dropdown">',
      '        <button class="btn btn-ghost btn-sm" @click="toggleWorkflowMenu">',
      '          {{ t(\'app.openWorkflow\') }} ▾',
      '        </button>',
      '        <div v-if="showWorkflowMenu" class="dropdown-menu">',
      '          <button class="dropdown-item"',
      '            v-for="wf in recentWorkflows" :key="wf.dir || wf.name"',
      '            @click="loadWorkflow(wf.dir || wf.name); showWorkflowMenu = false;">',
      '            {{ wf.name || wf.dir }}',
      '          </button>',
      '          <div v-if="recentWorkflows.length === 0" style="padding:8px 14px;color:var(--text-dim);font-size:12px;">',
      '            {{ t(\'app.noWorkflows\') }}',
      '          </div>',
      '        </div>',
      '      </div>',
      '      <button class="btn btn-blue btn-sm" @click="createNewWorkflow()">{{ t(\'app.newWorkflow\') }}</button>',
      '      <button v-if="workflowName" class="btn btn-ghost btn-sm" @click="saveWorkflow">💾 {{ t(\'app.save\') }}</button>',
      '    </div>',
      '  </header>',

      '  <div v-if="!workflowName" class="main-content">',
      '    <div class="welcome-screen">',
      '      <div class="welcome-icon">🔄</div>',
      '      <div class="welcome-title">{{ t(\'app.welcomeTitle\') }}</div>',
      '      <div class="welcome-desc">',
      '        {{ t(\'app.welcomeDesc1\') }}',
      '        {{ t(\'app.welcomeDesc2\') }}',
      '      </div>',
      '      <div class="welcome-actions">',
      '        <button class="btn btn-blue" @click="createNewWorkflow()">{{ t(\'app.newWorkflow\') }}</button>',
      '        <button class="btn btn-ghost" @click="toggleWorkflowMenu(); loadRecentWorkflows();">{{ t(\'app.openWorkflow\') }}</button>',
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
      '            {{ selectedStage.label || selectedStage.id }}',
      '          </span>',
      '          <div class="config-panel-actions">',
      '            <button class="btn btn-ghost btn-sm" @click="saveWorkflow">💾 {{ t(\'app.saveConfig\') }}</button>',
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
      '          :workflowStages="workflowData.stages"',
      '          :currentStageId="selectedStageId"',
      '          @update:modelValue="updateStageConfig" />',
      '      </div>',
      '      <div v-if="!selectedStage && workflowName" class="config-panel-content">',
      '        <div class="empty-state" style="margin-top:80px;">',
      '          {{ t(\'app.selectStage\') }}',
      '        </div>',
      '      </div>',
      '    </div>',
      '  </div>',

      '  <div v-if="workflowName" class="bottom-panel">',
      '    <div class="bottom-panel-header">',
      '      <span class="bottom-panel-title">{{ t(\'app.log\') }}</span>',
      '      <div class="log-tab-bar">',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'system\' }" @click="activeLogTab = \'system\'">{{ t(\'app.systemLog\') }}</button>',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'script\' }" @click="activeLogTab = \'script\'">{{ t(\'app.scriptOutput\') }} <span v-if="flatScriptLogCount" style="opacity:0.6;">({{ flatScriptLogCount }})</span></button>',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'history\' }" @click="activeLogTab = \'history\'; loadRunHistory()">{{ t(\'app.runHistory\') }}</button>',
      '      </div>',
      '      <span style="font-size:11px;color:var(--text-dim);">{{ completedStages }}/{{ totalStages }} {{ t(\'app.stages\') }}</span>',
      '    </div>',
      '    <div v-if="(isRunning || completedStages > 0) && activeLogTab !== \'history\'" class="progress-bar-container">',
      '      <div class="progress-bar-track">',
      '        <div class="progress-bar-fill stage-progress" :class="progressStatus" :style="{ width: overallProgress + \'%\' }"></div>',
      '      </div>',
      '      <div class="progress-label">{{ t(\'app.stages\') }} {{ overallProgress }}%</div>',
      '    </div>',
      '    <div v-if="scriptProgress.active && activeLogTab === \'script\'" class="script-progress-section">',
      '      <div class="progress-bar-track">',
      '        <div class="progress-bar-fill script-progress" :style="{ width: scriptProgress.pct + \'%\' }"></div>',
      '      </div>',
      '      <div class="script-status-line">',
      '        <span>{{ scriptProgress.current }}/{{ scriptProgress.total }}</span>',
      '        <span v-if="scriptProgress.eta">[{{ scriptProgress.elapsed }}&lt;{{ scriptProgress.eta }}, {{ scriptProgress.rate }}]</span>',
      '        <span v-for="(v, k) in scriptProgress.metrics" :key="k" class="metric-badge">{{ k }}={{ v }}</span>',
      '      </div>',
      '    </div>',
      '    <div class="log-viewer">',
      '      <template v-if="activeLogTab === \'system\'">',
      '        <div v-for="(line, i) in logLines" :key="i" class="log-line" :class="line.cls">',
      '          [{{ line.ts }}] {{ line.text }}',
      '        </div>',
      '        <div v-if="logLines.length === 0" style="color:var(--text-dim);font-style:italic;">{{ t(\'app.waitingToRun\') }}</div>',
      '      </template>',
      '      <template v-else-if="activeLogTab === \'script\'">',
      '        <div v-if="scriptStageIds.length > 1" style="display:flex;align-items:center;gap:6px;padding-bottom:4px;">',
      '          <select v-model="activeScriptStage" class="stage-select">',
      '            <option :value="null">{{ t(\'app.allStages\') }}</option>',
      '            <option v-for="sid in scriptStageIds" :key="sid" :value="sid">{{ sid }}</option>',
      '          </select>',
      '          <span style="font-size:11px;color:var(--text-dim);">{{ filteredScriptLogs.length }} {{ t(\'app.lines\') }}</span>',
      '        </div>',
      '        <div v-for="(line, i) in filteredScriptLogs" :key="line._id" class="log-line script">',
      '          <span v-if="line.stage_id && activeScriptStage === null" class="log-stage-tag">[{{ line.stage_id }}]</span>',
      '          <span v-if="line.ts" class="log-ts-tag">[{{ line.ts }}]</span>',
      '          {{ line.text }}',
      '        </div>',
      '        <div v-if="filteredScriptLogs.length === 0 && !scriptProgress.active" style="color:var(--text-dim);font-style:italic;">{{ t(\'app.noScriptOutput\') }}</div>',
      '        <div v-if="filteredScriptLogs.length === 0 && scriptProgress.active" class="log-line script" style="color:var(--text-dim);">{{ scriptProgress.rawLine }}</div>',
      '      </template>',
      '      <template v-if="activeLogTab === \'history\'">',
      '        <div v-if="runHistory.length === 0" style="color:var(--text-dim);font-style:italic;padding:8px 0;">{{ t(\'app.noRunHistory\') }}</div>',
      '        <div v-for="run in runHistory" :key="run.id" class="history-record">',
      '          <div class="history-time">{{ run.created_at ? run.created_at.replace(\'T\', \' \').substring(0, 16) : run.id }}</div>',
      '          <div class="history-status" :class="\'status-\' + run.status">',
      '            <span>{{ run.status === \'ok\' ? t(\'app.done\') : run.status === \'stopped\' ? t(\'app.stopped\') : run.status === \'error\' ? t(\'app.failed\') : run.status === \'running\' ? t(\'app.running\') : \'❓ \' + run.status }}</span>',
      '          </div>',
      '          <div v-if="run.stages && run.stages.length" class="history-stage-chain">',
      '            <template v-for="(s, si) in run.stages">',
      '              <div class="chain-node" :class="\'chain-\' + (s.status === \'ok\' ? \'done\' : s.status === \'running\' ? \'running\' : s.status === \'error\' || s.status === \'config_error\' ? \'error\' : s.status === \'stopped\' ? \'stopped\' : \'pending\')" :title="s.id + \': \' + s.status">',
      '                <span class="chain-icon">{{ s.status === \'ok\' ? \'●\' : s.status === \'running\' ? \'◑\' : s.status === \'error\' || s.status === \'config_error\' ? \'✕\' : s.status === \'stopped\' ? \'■\' : \'○\' }}</span>',
      '                <span class="chain-label">{{ s.id }}</span>',
      '              </div>',
      '              <span v-if="si < run.stages.length - 1" class="chain-arrow">→</span>',
      '            </template>',
      '          </div>',
      '          <div class="history-actions">',
      '            <button class="btn btn-ghost btn-xs" @click="viewRunLog(run.id)" :title="t(\'app.viewLog\')">{{ t(\'app.viewLog\') }}</button>',
      '            <button class="btn btn-ghost btn-xs" @click="openRunDir(run.id)" :title="t(\'app.openInFileManager\')">📂</button>',
      '          </div>',
      '        </div>',
      '      </template>',
      '    </div>',
      '  </div>',

      '  <div v-if="showLogModal" class="modal-overlay" @click.self="closeLogModal">',
      '    <div class="modal-content">',
      '      <div class="modal-header">',
      '        <span style="font-size:13px;font-weight:600;">{{ logModalTitle }}</span>',
      '        <button class="modal-close" @click="closeLogModal">&times;</button>',
      '      </div>',
      '      <div class="modal-body">',
      '        <div v-if="logModalLoading" style="color:var(--text-dim);font-style:italic;padding:12px 0;">{{ t(\'app.loading\') }}</div>',
      '        <div v-else class="log-content">',
      '          <div v-for="(line, i) in logModalLines" :key="i" class="log-line">{{ line }}</div>',
      '          <div v-if="logModalLines.length === 0" style="color:var(--text-dim);font-style:italic;">{{ t(\'app.logEmpty\') }}</div>',
      '        </div>',
      '      </div>',
      '    </div>',
      '  </div>',
      '',
      '  <div v-if="showSettingsModal" class="modal-overlay" @click.self="closeSettings">',
      '    <div class="modal-content" style="max-width:520px;">',
      '      <div class="modal-header">',
      '        <span style="font-size:13px;font-weight:600;">{{ t(\'app.settingsTitle\') }}</span>',
      '        <button class="modal-close" @click="closeSettings">&times;</button>',
      '      </div>',
      '      <div class="modal-body">',
      '        <div v-if="settingsLoading" style="color:var(--text-dim);font-style:italic;padding:12px 0;">{{ t(\'app.loading\') }}</div>',
      '        <div v-if="!settingsLoading">',
      '          <div class="schema-group">',
      '            <div class="schema-group-header">',
      '              <span class="schema-group-title">{{ t(\'app.workflowsRoot\') }}</span>',
      '            </div>',
      '            <div class="schema-group-body">',
      '              <div class="form-group">',
      '                <label class="form-label">{{ t(\'app.workflowsRootLabel\') }}</label>',
      '                <input class="form-input" type="text" v-model="settingsData.workflows_root" :placeholder="t(\'app.workflowsRootPlaceholder\')" />',
      '              </div>',
      '            </div>',
      '          </div>',
      '          <div class="schema-group">',
      '            <div class="schema-group-header">',
      '              <span class="schema-group-title">{{ t(\'app.modelPaths\') }}</span>',
      '            </div>',
      '            <div class="schema-group-body">',
      '              <div class="form-group">',
      '                <label class="form-label">{{ t(\'app.ditModel\') }}</label>',
      '                <input class="form-input" type="text" v-model="settingsData.pretrained_model_name_or_path" :placeholder="t(\'app.workflowsRootPlaceholder\')" />',
      '              </div>',
      '              <div class="form-group">',
      '                <label class="form-label">{{ t(\'app.textEncoder\') }}</label>',
      '                <input class="form-input" type="text" v-model="settingsData.qwen3" :placeholder="t(\'app.workflowsRootPlaceholder\')" />',
      '              </div>',
      '              <div class="form-group">',
      '                <label class="form-label">{{ t(\'app.vaeModel\') }}</label>',
      '                <input class="form-input" type="text" v-model="settingsData.vae" :placeholder="t(\'app.workflowsRootPlaceholder\')" />',
      '              </div>',
      '            </div>',
      '          </div>',
      '          <div class="schema-group">',
      '            <div class="schema-group-header">',
      '              <span class="schema-group-title">{{ t(\'app.hardwareSettings\') }}</span>',
      '            </div>',
      '            <div class="schema-group-body">',
      '              <div class="form-group">',
      '                <label class="form-label">{{ t(\'app.mixedPrecision\') }}</label>',
      '                <select class="form-select" v-model="settingsData.mixed_precision">',
      '                  <option value="bf16">bf16</option>',
      '                  <option value="fp16">fp16</option>',
      '                  <option value="fp32">fp32</option>',
      '                  <option value="no">no</option>',
      '                </select>',
      '              </div>',
      '              <div class="form-group">',
      '                <label class="form-label">{{ t(\'app.attnMode\') }}</label>',
      '                <select class="form-select" v-model="settingsData.attn_mode">',
      '                  <option value="flex">flex</option>',
      '                  <option value="sdpa">sdpa</option>',
      '                  <option value="flash">flash</option>',
      '                  <option value="xformers">xformers</option>',
      '                </select>',
      '              </div>',
      '            </div>',
      '          </div>',
      '          <div style="display:flex;justify-content:flex-end;gap:8px;padding-top:12px;">',
      '            <button class="btn btn-ghost btn-sm" @click="closeSettings">{{ t(\'app.cancel\') }}</button>',
      '            <button class="btn btn-blue btn-sm" @click="saveSettings" :disabled="settingsSaving">',
      '              {{ settingsSaving ? t(\'app.saving\') : t(\'app.saveSettings\') }}',
      '            </button>',
      '          </div>',
      '        </div>',
      '      </div>',
      '    </div>',
      '  </div>',
      '',
      '  <div v-if="showModal" class="modal-overlay">',
      '    <div class="modal-dialog">',
      '      <div class="modal-title">{{ modalTitle }}</div>',
      '      <input class="modal-input form-input" v-model="modalInput" :placeholder="modalPlaceholder" @keydown.enter="modalConfirm" @keydown.escape="modalCancel" ref="modalInputEl" autofocus />',
      '      <div class="modal-actions">',
      '        <button class="btn btn-ghost btn-sm" @click="modalCancel">{{ t(\'app.cancel\') }}</button>',
      '        <button class="btn btn-blue btn-sm" @click="modalConfirm">{{ t(\'app.ok\') }}</button>',
      '      </div>',
      '    </div>',
      '  </div>',
      '  <div v-for="toast in toasts" :key="toast.id" class="toast" :class="\'toast-\' + toast.type">',
      '    {{ toast.msg }}',
      '  </div>',

      '</div>',
    ].join("\n"),
  });

  app.config.globalProperties.t = window.t;

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

  I18n.init().then(function () {
    document.documentElement.lang = I18n.getLocale();
    currentLang.value = I18n.getLocale();
    app.mount("#app");
  });

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
