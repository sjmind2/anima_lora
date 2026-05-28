(function () {
  window.VueComponents = window.VueComponents || {};

  var METHODS = [
    { value: "train_lora", label: "LoRA", baseType: "lora" },
    { value: "train_loha", label: "LoHA", baseType: "lycoris" },
    { value: "train_locon", label: "LoCON", baseType: "lycoris" },
    { value: "train_lokr", label: "LoKR", baseType: "lycoris" },
  ];

  var COMBO_SWITCHES = [
    { key: "use_ortho", label: "Ortho" },
    { key: "use_moe_style", label: "MoE" },
    { key: "use_timestep_mask", label: "T-LoRA" },
    { key: "add_reft", label: "ReFT" },
  ];

  window.VueComponents.MethodSelector = {
    props: {
      modelValue: { type: String, default: "train_lora" },
      combos: { type: Object, default: function () { return {}; } },
    },
    emits: ["update:modelValue", "update:combos", "change"],
    data: function () {
      return {
        methods: METHODS,
        comboSwitches: COMBO_SWITCHES,
      };
    },
    computed: {
      currentMethod: function () {
        var self = this;
        return this.methods.find(function (m) {
          return m.value === self.modelValue;
        }) || this.methods[0];
      },
      isLora: function () {
        return this.currentMethod.baseType === "lora";
      },
    },
    methods: {
      onMethodChange: function (e) {
        this.$emit("update:modelValue", e.target.value);
        this.$emit("change", e.target.value);
      },
      toggleCombo: function (key) {
        var updated = Object.assign({}, this.combos);
        updated[key] = !updated[key];
        this.$emit("update:combos", updated);
        this.$emit("change", this.modelValue);
      },
    },
    template: [
      '<div class="method-selector">',
      '  <div class="method-selector-header">训练方法</div>',
      '  <div class="method-base-select">',
      '    <select class="form-select" :value="modelValue" @change="onMethodChange">',
      '      <option v-for="m in methods" :key="m.value" :value="m.value">{{ m.label }}</option>',
      '    </select>',
      '  </div>',
      '  <div v-if="isLora" class="combo-switches">',
      '    <div v-for="sw in comboSwitches" :key="sw.key"',
      '      class="combo-switch"',
      '      :class="{ active: !!combos[sw.key] }"',
      '      @click="toggleCombo(sw.key)">',
      '      <span class="switch-label">{{ sw.label }}</span>',
      '    </div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
