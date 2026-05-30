(function () {
  window.VueComponents = window.VueComponents || {};

  window.VueComponents.InfraSettings = {
    props: {
      modelValue: {
        type: Object,
        default: function () { return {}; },
      },
      workflowName: {
        type: String,
        default: "",
      },
    },
    emits: ["update:modelValue"],
    data: function () {
      return {
        schema: null,
        loading: false,
        saving: false,
        error: null,
        editData: {},
      };
    },
    watch: {
      modelValue: {
        handler: function (val) {
          this.editData = JSON.parse(JSON.stringify(val || {}));
        },
        immediate: true,
        deep: true,
      },
    },
    computed: {
      modelFields: function () {
        if (!this.schema || !this.schema.model_paths) return [];
        return Object.keys(this.schema.model_paths).map(function (key) {
          return { key: key, label: key, type: "text" };
        });
      },
      hardwareFields: function () {
        if (!this.schema || !this.schema.hardware) return [];
        return Object.keys(this.schema.hardware).map(function (key) {
          return { key: key, label: key, type: "text" };
        });
      },
    },
    mounted: function () {
      this.loadSchema();
    },
    methods: {
      loadSchema: function () {
        var self = this;
        this.loading = true;
        AnimaAPI.getSchema("infrastructure")
          .then(function (schema) {
            self.schema = schema;
          })
          .catch(function (err) {
            self.error = "加载基础设施配置失败: " + (err.error || err);
          })
          .finally(function () {
            self.loading = false;
          });
      },
      save: function () {
        var self = this;
        if (!this.workflowName) return;
        this.saving = true;
        AnimaAPI.setInfra(this.workflowName, this.editData)
          .then(function () {
            self.$emit("update:modelValue", JSON.parse(JSON.stringify(self.editData)));
          })
          .catch(function (err) {
            self.error = "保存失败: " + (err.error || err);
          })
          .finally(function () {
            self.saving = false;
          });
      },
      updateField: function (section, key, value) {
        if (!this.editData[section]) {
          this.editData[section] = {};
        }
        this.editData[section][key] = value;
      },
      getFieldValue: function (section, key) {
        if (this.editData[section]) return this.editData[section][key] || "";
        if (this.modelValue[section]) return this.modelValue[section][key] || "";
        return "";
      },
    },
    template: [
      '<div class="infra-settings">',
      '  <div v-if="loading" class="is-loading">加载中...</div>',
      '  <div v-if="error" class="is-error">{{ error }}</div>',
      '  <div v-if="!loading && !error" class="is-content">',
      '    <div v-if="modelFields.length > 0" class="schema-group">',
      '      <div class="schema-group-header">',
      '        <span class="schema-group-title">📁 {{ t(\'infraSettings.modelPaths\') }}</span>',
      '      </div>',
      '      <div class="schema-group-body">',
      '        <div v-for="f in modelFields" :key="f.key" class="form-group">',
      '          <label class="form-label">{{ f.label }}</label>',
      '          <input class="form-input" type="text"',
      '            :value="getFieldValue(\'model_paths\', f.key)"',
      '            @input="updateField(\'model_paths\', f.key, $event.target.value)" />',
      '        </div>',
      '      </div>',
      '    </div>',
      '    <div v-if="hardwareFields.length > 0" class="schema-group">',
      '      <div class="schema-group-header">',
      '        <span class="schema-group-title">⚙ {{ t(\'infraSettings.hardwareSettings\') }}</span>',
      '      </div>',
      '      <div class="schema-group-body">',
      '        <div v-for="f in hardwareFields" :key="f.key" class="form-group">',
      '          <label class="form-label">{{ f.label }}</label>',
      '          <input class="form-input" type="text"',
      '            :value="getFieldValue(\'hardware\', f.key)"',
      '            @input="updateField(\'hardware\', f.key, $event.target.value)" />',
      '        </div>',
      '      </div>',
      '    </div>',
      '    <div class="is-actions">',
      '      <button class="btn btn-blue" @click="save" :disabled="saving">',
      '        {{ saving ? "保存中..." : "💾 保存基础设施设置" }}',
      '      </button>',
      '    </div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
