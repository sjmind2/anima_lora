(function () {
  window.VueComponents = window.VueComponents || {};

  window.VueComponents.SchemaForm = {
    props: {
      schemaNames: { type: Array, default: function () { return []; } },
      modelValue: { type: Object, default: function () { return {}; } },
      workflowStages: { type: Array, default: function () { return []; } },
      currentStageId: { type: String, default: "" },
    },
    emits: ["update:modelValue"],
    data: function () {
      return {
        mergedGroups: [],
        loading: false,
        error: null,
        collapsedGroups: {},
        loadedCount: 0,
      };
    },
    watch: {
      schemaNames: {
        immediate: true,
        deep: true,
        handler: function (names) {
          if (names && names.length > 0) {
            var key = names.join(",");
            if (this._lastSchemaKey === key) return;
            this._lastSchemaKey = key;
            this.loadSchemas(names);
          }
        },
      },
    },
    computed: {
      groups: function () {
        return this.mergedGroups;
      },
    },
    methods: {
      loadSchemas: function (names) {
        var self = this;
        self.loading = true;
        self.error = null;
        self.mergedGroups = [];
        self.loadedCount = 0;

        var promises = names.map(function (name) {
          return AnimaAPI.getSchema(name).catch(function (err) {
            return { error: err.error || "Failed to load " + name, groups: [] };
          });
        });

        Promise.all(promises).then(function (results) {
          var allGroups = [];
          var hasError = false;

          results.forEach(function (data) {
            if (data.error) {
              hasError = true;
              self.error = data.error;
              return;
            }
            if (data.groups) {
              data.groups.forEach(function (g) {
                allGroups.push(g);
                if (g.collapsed) {
                  self.collapsedGroups[g.name] = true;
                }
              });
            }
          });

          self.mergedGroups = allGroups;
          self.loading = false;
          if (!hasError) {
            self.applyDefaults();
          }
        });
      },
      applyDefaults: function () {
        var updated = Object.assign({}, this.modelValue);
        var changed = false;
        this.mergedGroups.forEach(function (group) {
          if (!group.fields) return;
          group.fields.forEach(function (field) {
            if (
              field.default !== undefined &&
              (updated[field.key] === undefined || updated[field.key] === null || updated[field.key] === "")
            ) {
              updated[field.key] = field.default;
              changed = true;
            }
          });
        });
        if (changed) {
          this.$emit("update:modelValue", updated);
        }
      },
      toggleGroup: function (groupName) {
        this.collapsedGroups[groupName] = !this.collapsedGroups[groupName];
      },
      isGroupCollapsed: function (groupName) {
        return !!this.collapsedGroups[groupName];
      },
      updateField: function (key, value) {
        this.modelValue[key] = value;
        this.$emit("update:modelValue", this.modelValue);
      },
    },
    template: [
      '<div class="schema-form">',
      '  <div v-if="loading" class="empty-state">{{ t(\'schemaForm.loadingSchema\') }}</div>',
      '  <div v-if="error" class="empty-state" style="color: var(--red);">{{ error }}</div>',
      '  <div v-if="!loading && !error && groups.length > 0">',
      '    <div v-for="group in groups" :key="group.name" class="schema-group">',
      '      <div class="schema-group-header" @click="toggleGroup(group.name)">',
      '        <span class="schema-group-title">',
      '          <span class="schema-group-toggle" :class="{ collapsed: isGroupCollapsed(group.name) }">▼</span>',
      '          {{ group.label || group.name }}',
      '        </span>',
      '      </div>',
      '      <div class="schema-group-body" :class="{ collapsed: isGroupCollapsed(group.name) }">',
      '        <field-renderer',
      '          v-for="field in (group.fields || [])"',
      '          :key="group.name + \'.\' + field.key"',
      '          :field="field"',
      '          :modelValue="modelValue[field.key]"',
      '          :allValues="modelValue"',
      '          :workflowStages="workflowStages"',
      '          :currentStageId="currentStageId"',
      '          @update:modelValue="updateField(field.key, $event)" />',
      '      </div>',
      '    </div>',
      '  </div>',
      '</div>',
    ].join("\n"),
  };
})();
