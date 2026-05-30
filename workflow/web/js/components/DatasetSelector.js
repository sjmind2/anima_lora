(function () {
  window.VueComponents = window.VueComponents || {};

  function inferNumRepeats(name) {
    var m = name.match(/^(\d+)[_]/);
    return m ? parseInt(m[1], 10) : 1;
  }

  window.VueComponents.DatasetSelector = {
    props: {
      availableDatasets: {
        type: Array,
        default: function () { return []; },
      },
      modelValue: {
        type: Array,
        default: function () { return []; },
      },
    },
    emits: ["update:modelValue"],
    data: function () {
      return {
        expandedStages: {},
        repeatOverrides: {},
      };
    },
    computed: {
      selectedRefs: function () {
        return this.modelValue || [];
      },
    },
    methods: {
      toggleStage: function (stageId) {
        this.expandedStages[stageId] = !this.expandedStages[stageId];
      },
      isExpanded: function (stageId) {
        return !!this.expandedStages[stageId];
      },
      isSubsetSelected: function (ref) {
        return this.selectedRefs.indexOf(ref) >= 0;
      },
      toggleSubset: function (ref, stage) {
        var idx = this.selectedRefs.indexOf(ref);
        var next = this.selectedRefs.slice();
        if (idx >= 0) {
          next.splice(idx, 1);
        } else {
          next.push(ref);
        }
        this.$emit("update:modelValue", next);
      },
      getNumRepeats: function (subsetName) {
        if (this.repeatOverrides[subsetName] !== undefined) {
          return this.repeatOverrides[subsetName];
        }
        return inferNumRepeats(subsetName);
      },
      setNumRepeats: function (subsetName, val) {
        var n = parseInt(val, 10);
        if (isNaN(n) || n < 1) n = 1;
        this.repeatOverrides[subsetName] = n;
      },
      getSubsetRef: function (stage, subset) {
        if (subset.ref) return subset.ref;
        return stage.stage_id + "/" + (subset.name || subset.dir || "");
      },
    },
    template: [
      '<div class="dataset-selector">',
      '  <div v-if="availableDatasets.length === 0" class="empty-state">',
      '    暂无可用数据集',
      '  </div>',
      '  <div v-for="stage in availableDatasets" :key="stage.stage_id" class="ds-stage-group">',
      '    <div class="ds-stage-header" @click="toggleStage(stage.stage_id)">',
      '      <span class="ds-toggle" :class="{ expanded: isExpanded(stage.stage_id) }">▶</span>',
      '      <span class="ds-stage-name">{{ stage.stage_id }}</span>',
      '      <span class="ds-stage-count">{{ stage.subsets ? stage.subsets.length : 0 }} 子集</span>',
      '    </div>',
      '    <div v-if="isExpanded(stage.stage_id)" class="ds-subset-list">',
      '      <div v-for="subset in (stage.subsets || [])" :key="getSubsetRef(stage, subset)" class="ds-subset-row">',
      '        <label class="ds-checkbox-row">',
      '          <input type="checkbox"',
      '            :checked="isSubsetSelected(getSubsetRef(stage, subset))"',
      '            @change="toggleSubset(getSubsetRef(stage, subset), stage)" />',
      '          <span class="ds-subset-name">{{ subset.name || subset.dir || "unnamed" }}</span>',
      '        </label>',
      '        <label class="ds-repeat-control">',
      '          <span class="ds-repeat-label">{{ t(\'datasetSelector.repeat\') }}</span>',
      '          <input type="number" class="form-input ds-repeat-input"',
      '            :value="getNumRepeats(subset.name || subset.dir || \'\')"',
      '            @input="setNumRepeats((subset.name || subset.dir || \'\'), $event.target.value)"',
      '            min="1" max="100" />',
      '        </label>',
      '      </div>',
      '    </div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
