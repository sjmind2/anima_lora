(function () {
  window.VueComponents = window.VueComponents || {};

  window.VueComponents.RunControl = {
    props: {
      runState: {
        type: Object,
        default: function () {
          return {
            status: "idle",
            totalStages: 0,
            completedStages: 0,
            currentStage: null,
            progress: 0,
          };
        },
      },
    },
    emits: ["run", "stop"],
    computed: {
      isRunning: function () {
        return this.runState.status === "running";
      },
      overallPercent: function () {
        if (!this.runState.totalStages) return 0;
        return Math.round(
          (this.runState.completedStages / this.runState.totalStages) * 100
        );
      },
      barClass: function () {
        var s = this.runState.status;
        if (s === "done") return "done";
        if (s === "failed") return "failed";
        if (s === "running") return "running";
        return "";
      },
      stageItems: function () {
        var items = [];
        if (!this.runState.stages) return items;
        for (var i = 0; i < this.runState.stages.length; i++) {
          var s = this.runState.stages[i];
          items.push({
            id: s.id || i,
            name: s.name || ("Stage " + (i + 1)),
            status: s.status || "pending",
            progress: s.progress || 0,
          });
        }
        return items;
      },
    },
    methods: {
      handleRun: function () {
        this.$emit("run");
      },
      handleStop: function () {
        this.$emit("stop");
      },
      miniBarClass: function (status) {
        if (status === "done") return "rc-mini-done";
        if (status === "failed") return "rc-mini-failed";
        if (status === "running") return "rc-mini-running";
        return "";
      },
    },
    template: [
      '<div class="run-control">',
      '  <div class="rc-btn-row">',
      '    <button class="btn btn-green" @click="handleRun" :disabled="isRunning">',
      '      {{ t(\'runControl.run\') }}',
      '    </button>',
      '    <button class="btn btn-red" @click="handleStop" :disabled="!isRunning">',
      '      {{ t(\'runControl.stop\') }}',
      '    </button>',
      '  </div>',
      '  <div class="rc-progress-section">',
      '    <div class="rc-progress-label">',
      '      <span>{{ runState.completedStages || 0 }} / {{ runState.totalStages || 0 }} {{ t(\'runControl.stagesLabel\') }}</span>',
      '      <span>{{ overallPercent }}%</span>',
      '    </div>',
      '    <div class="progress-bar-track">',
      '      <div class="progress-bar-fill" :class="barClass" :style="{ width: overallPercent + \'%\' }"></div>',
      '    </div>',
      '  </div>',
      '  <div v-if="stageItems.length > 0" class="rc-stage-indicators">',
      '    <div v-for="s in stageItems" :key="s.id" class="rc-mini-stage">',
      '      <div class="rc-mini-label">{{ s.name }}</div>',
      '      <div class="rc-mini-bar-track">',
      '        <div class="rc-mini-bar-fill" :class="miniBarClass(s.status)"',
      '          :style="{ width: s.progress + \'%\' }"></div>',
      '      </div>',
      '    </div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
