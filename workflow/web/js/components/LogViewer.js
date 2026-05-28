(function () {
  window.VueComponents = window.VueComponents || {};

  var MAX_LINES = 5000;

  var LEVEL_COLORS = {
    INFO: "var(--blue)",
    WARN: "var(--orange)",
    ERROR: "var(--red)",
    SUCCESS: "var(--green)",
  };

  window.VueComponents.LogViewer = {
    props: {
      logs: {
        type: Array,
        default: function () { return []; },
      },
    },
    data: function () {
      return {
        paused: false,
        searchText: "",
        stageFilter: "",
        viewerEl: null,
      };
    },
    computed: {
      stages: function () {
        var seen = {};
        var result = [];
        this.logs.forEach(function (log) {
          if (log.stage_id && !seen[log.stage_id]) {
            seen[log.stage_id] = true;
            result.push(log.stage_id);
          }
        });
        return result;
      },
      filteredLogs: function () {
        var search = this.searchText.toLowerCase();
        var stage = this.stageFilter;
        var result = this.logs;
        if (stage) {
          result = result.filter(function (log) {
            return log.stage_id === stage;
          });
        }
        if (search) {
          result = result.filter(function (log) {
            return (log.message || "").toLowerCase().indexOf(search) >= 0;
          });
        }
        if (result.length > MAX_LINES) {
          result = result.slice(result.length - MAX_LINES);
        }
        return result;
      },
    },
    watch: {
      logs: {
        handler: function () {
          if (!this.paused) {
            this.scrollToBottom();
          }
        },
        deep: true,
      },
    },
    mounted: function () {
      this.viewerEl = this.$refs.logContainer;
    },
    methods: {
      onScroll: function () {
        var el = this.viewerEl;
        if (!el) return;
        var atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
        this.paused = !atBottom;
      },
      scrollToBottom: function () {
        var self = this;
        this.$nextTick(function () {
          if (self.viewerEl) {
            self.viewerEl.scrollTop = self.viewerEl.scrollHeight;
          }
        });
      },
      resumeScroll: function () {
        this.paused = false;
        this.scrollToBottom();
      },
      levelClass: function (level) {
        if (!level) return "info";
        var l = level.toUpperCase();
        if (l === "ERROR") return "error";
        if (l === "WARN" || l === "WARNING") return "warn";
        if (l === "SUCCESS") return "success";
        return "info";
      },
      levelColor: function (level) {
        if (!level) return LEVEL_COLORS.INFO;
        return LEVEL_COLORS[level.toUpperCase()] || LEVEL_COLORS.INFO;
      },
      highlightText: function (text) {
        if (!this.searchText || !text) return text;
        var escaped = this.searchText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        var re = new RegExp("(" + escaped + ")", "gi");
        return text.replace(re, '<mark class="lv-highlight">$1</mark>');
      },
    },
    template: [
      '<div class="log-viewer-container">',
      '  <div class="lv-toolbar">',
      '    <input type="text" class="form-input lv-search" placeholder="搜索日志..."',
      '      v-model="searchText" />',
      '    <select class="form-select lv-stage-filter" v-model="stageFilter">',
      '      <option value="">全部阶段</option>',
      '      <option v-for="s in stages" :key="s" :value="s">{{ s }}</option>',
      '    </select>',
      '    <button v-if="paused" class="btn btn-ghost btn-sm" @click="resumeScroll">',
      '      ▼ 继续',
      '    </button>',
      '    <span v-if="paused" class="lv-paused-badge">已暂停</span>',
      '  </div>',
      '  <div class="lv-content" ref="logContainer" @scroll="onScroll">',
      '    <div v-for="(log, i) in filteredLogs" :key="i" class="lv-line" :class="levelClass(log.level)">',
      '      <span class="lv-ts">{{ log.timestamp }}</span>',
      '      <span class="lv-level" :style="{ color: levelColor(log.level) }">{{ log.level }}</span>',
      '      <span v-if="log.stage_id" class="lv-stage">[{{ log.stage_id }}]</span>',
      '      <span class="lv-msg" v-html="highlightText(log.message)"></span>',
      '    </div>',
      '    <div v-if="filteredLogs.length === 0" class="lv-empty">',
      '      暂无日志',
      '    </div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
