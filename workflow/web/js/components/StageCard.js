(function () {
  window.VueComponents = window.VueComponents || {};

  var TYPE_ICONS = {
    preprocess: "📁",
    train: "🎯",
  };

  var STATUS_ICONS = {
    pending: "⏳",
    starting: "▶",
    running: "🔄",
    done: "✅",
    failed: "❌",
    stopped: "⏹",
  };

  window.VueComponents.StageCard = {
    props: {
      stage: { type: Object, required: true },
      index: { type: Number, required: true },
      selected: { type: Boolean, default: false },
      runState: { type: Object, default: function () { return {}; } },
    },
    emits: ["select", "remove", "dragstart", "dragover", "drop", "dragend"],
    computed: {
      typeIcons: function () { return TYPE_ICONS; },
      statusIcon: function () {
        var st = this.stageStatus;
        return STATUS_ICONS[st] || "⏳";
      },
      stageStatus: function () {
        if (this.runState && this.runState[this.stage.id]) {
          return this.runState[this.stage.id].status || "pending";
        }
        return "pending";
      },
      stageProgress: function () {
        if (this.runState && this.runState[this.stage.id]) {
          return this.runState[this.stage.id].progress || 0;
        }
        return 0;
      },
      statusClass: function () {
        return this.stageStatus;
      },
      hasDependencies: function () {
        return this.stage.depends_on && this.stage.depends_on.length > 0;
      },
      depCount: function () {
        return this.stage.depends_on ? this.stage.depends_on.length : 0;
      },
      stageLabel: function () {
        if (this.stage.label) return this.stage.label;
        var typeLabel = this.stage.type === "train" ? "Train" : "Preprocess";
        return typeLabel + " " + (this.index + 1);
      },
    },
    methods: {
      onSelect: function () {
        this.$emit("select", this.stage.id);
      },
      onRemove: function (e) {
        e.stopPropagation();
        this.$emit("remove", this.stage.id);
      },
    },
    template: [
      '<div class="stage-card"',
      '  :class="[statusClass, { active: selected }]"',
      '  @click="onSelect"',
      '  draggable="true"',
      '  @dragstart="$emit(\'dragstart\', index, $event)"',
      '  @dragover.prevent="$emit(\'dragover\', index, $event)"',
      '  @drop="$emit(\'drop\', index, $event)"',
      '  @dragend="$emit(\'dragend\')">',
      '  <span class="drag-handle" :title="t(\'stageCard.dragToReorder\')">⠿</span>',
      '  <span class="stage-icon">{{ typeIcons[stage.type] || "📄" }}</span>',
      '  <div class="stage-info">',
      '    <div class="stage-name">{{ stageLabel }}</div>',
      '    <div class="stage-meta">',
      '      <span class="status-icon">{{ statusIcon }}</span>',
      '      <span v-if="hasDependencies" class="dep-badge" title="依赖">',
      '        ↗ {{ depCount }}',
      '      </span>',
      '    </div>',
      '  </div>',
      '  <button class="remove-btn" @click="onRemove" title="删除阶段">✕</button>',
      '  <div v-if="stageProgress > 0" class="mini-progress" :style="{ width: stageProgress + \'%\' }"></div>',
      '</div>',
    ].join("\n"),
  };
})();
