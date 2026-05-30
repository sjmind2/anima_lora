(function () {
  window.VueComponents = window.VueComponents || {};

  window.VueComponents.StageList = {
    props: {
      stages: { type: Array, default: function () { return []; } },
      selectedId: { type: String, default: null },
      runState: { type: Object, default: function () { return {}; } },
      isRunning: { type: Boolean, default: false },
    },
    emits: ["select", "add", "remove", "reorder", "run", "stop"],
    data: function () {
      return {
        showAddMenu: false,
        dragIndex: null,
        menuStyle: {},
      };
    },
    methods: {
      addStage: function (type) {
        this.showAddMenu = false;
        this.$emit("add", type);
      },
      toggleAddMenu: function (e) {
        this.showAddMenu = !this.showAddMenu;
        if (this.showAddMenu && e && e.target) {
          var rect = e.target.getBoundingClientRect();
          this.menuStyle = {
            position: "fixed",
            top: rect.bottom + 4 + "px",
            left: rect.left + "px",
            width: rect.width + "px",
            zIndex: 10000,
          };
        }
      },
      onDragStart: function (index, event) {
        this.dragIndex = index;
        event.dataTransfer.effectAllowed = "move";
      },
      onDragOver: function (index, event) {
        event.dataTransfer.dropEffect = "move";
      },
      onDrop: function (toIndex, event) {
        if (this.dragIndex !== null && this.dragIndex !== toIndex) {
          this.$emit("reorder", this.dragIndex, toIndex);
        }
        this.dragIndex = null;
      },
      onDragEnd: function () {
        this.dragIndex = null;
      },
      handleRun: function () {
        this.$emit("run");
      },
      handleStop: function () {
        this.$emit("stop");
      },
    },
    template: [
      '<div style="display:flex;flex-direction:column;height:100%;">',
      '  <div class="stage-panel-header">',
      '    <span>{{ t(\'stageList.stagePanel\') }}</span>',
      '  </div>',
      '  <div class="stage-list">',
      '    <stage-card',
      '      v-for="(stage, idx) in stages"',
      '      :key="stage.id"',
      '      :stage="stage"',
      '      :index="idx"',
      '      :selected="stage.id === selectedId"',
      '      :runState="runState"',
      '      @select="$emit(\'select\', $event)"',
      '      @remove="$emit(\'remove\', $event)"',
      '      @dragstart="onDragStart"',
      '      @dragover="onDragOver"',
      '      @drop="onDrop"',
      '      @dragend="onDragEnd" />',
      '    <div v-if="stages.length === 0" class="empty-state">',
      '      {{ t(\'stageList.noStages\') }}',
      '    </div>',
      '  </div>',
      '  <div class="add-stage-area">',
      '    <div class="dropdown" style="width:100%;">',
      '      <button class="btn btn-blue btn-sm" style="width:100%;" @click="toggleAddMenu">',
      '        {{ t(\'stageList.addStage\') }}',
      '      </button>',
      '      <div v-if="showAddMenu" class="dropdown-menu" :style="menuStyle">',
      '        <button class="dropdown-item" @click="addStage(\'preprocess\')">📁 Preprocess</button>',
      '        <button class="dropdown-item" @click="addStage(\'train\')">🎯 Train</button>',
      '      </div>',
      '    </div>',
      '  </div>',
      '  <div class="run-controls">',
      '    <div class="btn-row">',
      '      <button class="btn btn-green btn-sm" style="flex:1;" @click="handleRun" :disabled="isRunning">',
      '        {{ t(\'stageList.run\') }}',
      '      </button>',
      '      <button class="btn btn-red btn-sm" style="flex:1;" @click="handleStop" :disabled="!isRunning">',
      '        {{ t(\'stageList.stop\') }}',
      '      </button>',
      '    </div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
