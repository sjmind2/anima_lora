(function () {
  window.VueComponents = window.VueComponents || {};

  window.VueComponents.FieldRenderer = {
    props: {
      field: { type: Object, required: true },
      modelValue: { required: false },
      allValues: { type: Object, default: function () { return {}; } },
    },
    emits: ["update:modelValue"],
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
      '    <option v-for="opt in (field.choices || [])" :key="opt" :value="opt">{{ opt }}</option>',
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

      '  <div v-if="field.type === \'dataset_ref\'"',
      '    class="form-input"',
      '    style="color: var(--text-dim); cursor: default; font-style: italic;">',
      '    {{ currentValue || \'（将自动从上游阶段获取数据集）\' }}',
      '  </div>',

      '  <div v-if="field.type === \'checkpoint_ref\'"',
      '    style="display: flex; align-items: center; gap: 8px;">',
      '    <input type="text"',
      '      class="form-input"',
      '      :value="currentValue"',
      '      @input="$emit(\'update:modelValue\', $event.target.value)"',
      '      placeholder="留空不使用" />',
      '  </div>',

      '  <div v-if="field.type === \'list[str]\'"',
      '    style="display: flex; flex-wrap: wrap; gap: 6px;">',
      '    <label v-for="opt in (field.choices || [])" :key="opt"',
      '      class="combo-switch"',
      '      :class="{ active: (currentValue || []).includes(opt) }"',
      '      style="cursor: pointer;">',
      '      <input type="checkbox"',
      '        :checked="(currentValue || []).includes(opt)"',
      '        @change="toggleListItem(opt)"',
      '        style="display: none;" />',
      '      {{ opt }}',
      '    </label>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
