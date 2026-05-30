(function () {
  window.VueComponents = window.VueComponents || {};

  window.VueComponents.FieldRenderer = {
    props: {
      field: { type: Object, required: true },
      modelValue: { required: false },
      allValues: { type: Object, default: function () { return {}; } },
      workflowStages: { type: Array, default: function () { return []; } },
      currentStageId: { type: String, default: "" },
    },
    emits: ["update:modelValue"],
    data: function () {
      return {
        bucketStats: null,
        bucketStatsError: null,
        bucketStatsLoading: false,
      };
    },
    computed: {
      isHidden: function () {
        return this.field.hidden === true;
      },
      conditionMet: function () {
        return this.evaluateCondition(this.field.condition);
      },
      currentValue: {
        get: function () {
          if (this.modelValue !== undefined && this.modelValue !== null && this.modelValue !== "") {
            return this.modelValue;
          }
          if (this.field.default !== undefined) {
            return this.field.default;
          }
          if (this.field.type === "bool") return false;
          if (this.field.type === "int" || this.field.type === "float") return null;
          if (this.field.type === "enum") {
            if (this.field.choices && this.field.choices.length > 0) return this.field.choices[0];
            return "";
          }
          return "";
        },
        set: function (val) {
          this.$emit("update:modelValue", val);
        },
      },
      labelMarker: function () {
        if (this.field.auto_set) return "auto_set";
        if (this.field.conditional_required) return "conditional_required";
        if (this.field.required) return "required";
        return "";
      },
      upstreamPreprocessStages: function () {
        var self = this;
        return this.workflowStages.filter(function (s) {
          return s.type === "preprocess" && s.id !== self.currentStageId;
        });
      },
      upstreamTrainStages: function () {
        var self = this;
        return this.workflowStages.filter(function (s) {
          return s.type === "train" && s.id !== self.currentStageId;
        });
      },
    },
    methods: {
      evaluateCondition: function (cond) {
        if (!cond) return true;
        try {
          var parts = cond.split(/(==|!=|>=|<=|>|<)/);
          if (parts.length === 1) {
            var trimmed = cond.trim();
            if (trimmed === "true") return true;
            if (trimmed === "false") return false;
            var val = this.allValues[trimmed];
            return val === true || (val && val !== "false");
          }
          var left = parts[0].trim();
          var op = parts[1].trim();
          var right = parts[2].trim().replace(/^['"]|['"]$/g, "");
          var leftVal = this.allValues[left];
          if (leftVal === undefined) leftVal = left;
          switch (op) {
            case "==": return String(leftVal) === String(right);
            case "!=": return String(leftVal) !== String(right);
            case ">": return Number(leftVal) > Number(right);
            case "<": return Number(leftVal) < Number(right);
            case ">=": return Number(leftVal) >= Number(right);
            case "<=": return Number(leftVal) <= Number(right);
            default: return true;
          }
        } catch (e) {
          return true;
        }
      },
      handleNumberInput: function (e) {
        var raw = e.target.value;
        if (raw === "") {
          this.$emit("update:modelValue", null);
          return;
        }
        if (this.field.type === "int") {
          var v = parseInt(raw, 10);
          if (!isNaN(v)) this.$emit("update:modelValue", v);
        } else {
          var v2 = parseFloat(raw);
          if (!isNaN(v2)) this.$emit("update:modelValue", v2);
        }
      },
      handleToggle: function (e) {
        this.$emit("update:modelValue", e.target.checked);
      },
      toggleListItem: function (item) {
        var list = this.currentValue ? this.currentValue.slice() : [];
        var idx = list.indexOf(item);
        if (idx >= 0) {
          list.splice(idx, 1);
        } else {
          list.push(item);
        }
        this.$emit("update:modelValue", list);
      },
      analyzeBucketStats: function () {
        var self = this;
        var allValues = this.allValues || {};
        var sourceDir = allValues.source_image_dir || "";
        if (!sourceDir) return;
        var selected = this.currentValue || [];
        self.bucketStatsLoading = true;
        AnimaAPI.analyzeBucketStats(sourceDir, selected).then(function (result) {
          self.bucketStatsLoading = false;
          if (result.error) {
            self.bucketStatsError = result.error;
            self.bucketStats = null;
          } else {
            self.bucketStats = result;
            self.bucketStatsError = null;
          }
        }).catch(function () {
          self.bucketStatsLoading = false;
          self.bucketStatsError = "Request failed";
          self.bucketStats = null;
        });
      },
      isDatasetSelected: function (stageId) {
        var val = this.currentValue;
        if (!val || !Array.isArray(val)) return false;
        return val.indexOf(stageId) >= 0;
      },
      toggleDatasetRef: function (stageId) {
        var val = this.currentValue ? this.currentValue.slice() : [];
        var idx = val.indexOf(stageId);
        if (idx >= 0) {
          val.splice(idx, 1);
        } else {
          val.push(stageId);
        }
        this.$emit("update:modelValue", val);
      },
      selectCheckpoint: function (val) {
        this.$emit("update:modelValue", val);
      },
    },
    template: [
      '<div class="form-group" :class="{ \'field-hidden\': isHidden || !conditionMet }">',
      '  <label class="form-label">',
      '    <span v-if="labelMarker === \'required\'" class="required" title="必填">*</span>',
      '    <span v-if="labelMarker === \'conditional_required\'" class="conditional-required" title="条件必填">*</span>',
      '    <span v-if="labelMarker === \'auto_set\'" class="auto-set" title="自动设置">⚡</span>',
      '    {{ field.label || field.key }}',
      '    <span v-if="field.help" class="help-text">{{ field.help }}</span>',
      '  </label>',

      '  <input v-if="field.type === \'int\' || field.type === \'float\'"',
      '    type="number"',
      '    class="form-input"',
      '    :step="field.type === \'float\' ? \'any\' : \'1\'"',
      '    :value="currentValue"',
      '    @input="handleNumberInput"',
      '    :placeholder="field.default !== undefined ? String(field.default) : \'\'" />',

      '  <label v-if="field.type === \'bool\'" class="form-toggle">',
      '    <input type="checkbox"',
      '      :checked="!!currentValue"',
      '      @change="handleToggle" />',
      '    <span class="slider"></span>',
      '  </label>',

      '  <select v-if="field.type === \'enum\'"',
      '    class="form-select"',
      '    :value="currentValue"',
      '    @change="$emit(\'update:modelValue\', $event.target.value)">',
      '    <option v-for="opt in (field.choices || [])" :key="opt" :value="opt">{{ (field.choice_labels || {})[opt] || opt }}</option>',
      '  </select>',

      '  <input v-if="field.type === \'str\'"',
      '    type="text"',
      '    class="form-input"',
      '    :value="currentValue"',
      '    @input="$emit(\'update:modelValue\', $event.target.value)"',
      '    :placeholder="field.default !== undefined ? String(field.default) : \'\'" />',

      '  <input v-if="field.type === \'path\'"',
      '    type="text"',
      '    class="form-input"',
      '    :value="currentValue"',
      '    @input="$emit(\'update:modelValue\', $event.target.value)"',
      '    :placeholder="field.help || \'路径\'" />',

      '  <div v-if="field.type === \'dataset_ref\'" class="dataset-ref-field">',
      '    <div v-if="upstreamPreprocessStages.length === 0" class="form-input" style="color: var(--text-dim); font-style: italic;">',
      '      无上游预处理阶段',
      '    </div>',
      '    <div v-else class="dataset-ref-list">',
      '      <label v-for="ps in upstreamPreprocessStages" :key="ps.id" class="dataset-ref-item">',
      '        <input type="checkbox"',
      '          :checked="isDatasetSelected(ps.id)"',
      '          @change="toggleDatasetRef(ps.id)" />',
      '        <span class="dataset-ref-icon">📁</span>',
      '        <span>{{ ps.label || ps.id }}</span>',
      '      </label>',
      '    </div>',
      '  </div>',

      '  <div v-if="field.type === \'checkpoint_ref\'" class="checkpoint-ref-field">',
      '    <select v-if="upstreamTrainStages.length > 0"',
      '      class="form-select"',
      '      :value="currentValue || \'\'"',
      '      @change="selectCheckpoint($event.target.value)">',
      '      <option value="">不使用上游 checkpoint</option>',
      '      <option v-for="ts in upstreamTrainStages" :key="ts.id"',
      '        :value="\'${\' + ts.id + \'.safetensors_path}\'">',
      '        🎯 {{ ts.id }}',
      '      </option>',
      '    </select>',
      '    <input v-else type="text" class="form-input"',
      '      :value="currentValue"',
      '      @input="$emit(\'update:modelValue\', $event.target.value)"',
      '      placeholder="无上游训练阶段" />',
      '    <div v-if="currentValue" class="checkpoint-ref-preview">{{ currentValue }}</div>',
      '  </div>',

      '  <div v-if="field.type === \'list[str]\' && field.choice_details"',
      '    class="bucket-options-area">',
      '    <div v-if="field.help" style="display:flex;width:100%;margin-bottom:4px;align-items:center;">',
      '      <span style="font-size:11px;color:var(--text-dim);">{{ field.help }}</span>',
      '      <button class="bucket-analyze-btn"',
      '        :disabled="!allValues.source_image_dir || bucketStatsLoading"',
      '        @click="analyzeBucketStats">',
      '        {{ bucketStatsLoading ? "分析中..." : "分析数据集" }}',
      '      </button>',
      '    </div>',
      '    <div v-if="!field.help" style="display:flex;width:100%;margin-bottom:4px;justify-content:flex-end;">',
      '      <button class="bucket-analyze-btn"',
      '        :disabled="!allValues.source_image_dir || bucketStatsLoading"',
      '        @click="analyzeBucketStats">',
      '        {{ bucketStatsLoading ? "分析中..." : "分析数据集" }}',
      '      </button>',
      '    </div>',
      '    <div v-for="opt in (field.choices || [])" :key="opt"',
      '      class="bucket-option-item"',
      '      :class="{ active: (currentValue || []).includes(opt) }"',
      '      @click="toggleListItem(opt)">',
      '      <input type="checkbox"',
      '        :checked="(currentValue || []).includes(opt)"',
      '        style="display: none;" />',
      '      <div class="bucket-option-header">',
      '        <span>{{ (field.choice_labels || {})[opt] || opt }}</span>',
      '      </div>',
      '      <div v-if="field.choice_details[opt]" class="bucket-detail-line">',
      '        {{ field.choice_details[opt].resolutions.join("  ") }}',
      '      </div>',
      '      <div v-if="bucketStats && bucketStats.families[opt]" class="bucket-stats-line">',
      '        <span class="stats-original">原始: {{ bucketStats.families[opt].original }}张</span>',
      '        <span style="margin: 0 4px;">·</span>',
      '        <span class="stats-resized" :class="{ dim: !bucketStats.families[opt].resized }">',
      '          缩放后: {{ bucketStats.families[opt].resized }}张',
      '        </span>',
      '      </div>',
      '    </div>',
      '  </div>',
      '',
      '  <div v-if="field.type === \'list[str]\' && !field.choice_details"',
      '    style="display: flex; flex-wrap: wrap; gap: 6px;">',
      '    <label v-for="opt in (field.choices || [])" :key="opt"',
      '      class="combo-switch"',
      '      :class="{ active: (currentValue || []).includes(opt) }"',
      '      style="cursor: pointer; font-size: 11px;"',
      '      :title="(field.choice_labels || {})[opt] || opt">',
      '      <input type="checkbox"',
      '        :checked="(currentValue || []).includes(opt)"',
      '        @change="toggleListItem(opt)"',
      '        style="display: none;" />',
      '      {{ (field.choice_labels || {})[opt] || opt }}',
      '    </label>',
      '    <div v-if="field.help" style="font-size:11px;color:var(--text-dim);width:100%;margin-top:2px;">{{ field.help }}</div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
