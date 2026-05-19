const NODE_CONFIG = {
    "XY Input (Anima): Sampler/Scheduler": {
        countWidget: "input_count",
        baseNames: ["sampler_", "scheduler_"],
        max: 10
    },
    "XY Input (Anima): Positive Prompt S/R": {
        countWidget: "replace_count",
        baseNames: ["replace_"],
        max: 10
    },
    "XY Input (Anima): Negative Prompt S/R": {
        countWidget: "replace_count",
        baseNames: ["replace_"],
        max: 10
    },
    "XY Input (Anima): Anima Adapter": {
        countWidget: "input_count",
        baseNames: ["adapter_", "strength_lora_", "strength_reft_"],
        max: 10
    },
    "XY Input (Anima): Anima Postfix": {
        countWidget: "input_count",
        baseNames: ["postfix_", "strength_"],
        max: 10
    },
    "XY Input (Anima): Checkpoint": {
        countWidget: "input_count",
        baseNames: ["unet_name_"],
        max: 10
    },
    "XY Input (Anima): VAE": {
        countWidget: "input_count",
        baseNames: ["vae_name_"],
        max: 10
    },
    "XY Input (Anima): LoRA": {
        countWidget: "input_count",
        baseNames: ["lora_name_", "model_strength_", "clip_strength_"],
        max: 10
    }
};

function toggleWidget(node, widget, show) {
    if (widget._origType === undefined) {
        widget._origType = widget.type;
        widget._origComputeSize = widget.computeSize;
    }
    if (show) {
        widget.type = widget._origType;
        widget.computeSize = widget._origComputeSize;
    } else {
        widget.type = "anima_xy_hide";
        widget.computeSize = () => [0, -4];
    }
    const newSize = node.computeSize();
    node.setSize([node.size[0], newSize[1]]);
}

function updateVisibility(node, config) {
    const countWidget = node.widgets.find(w => w.name === config.countWidget);
    if (!countWidget) return;
    const count = countWidget.value;
    for (let i = 1; i <= config.max; i++) {
        for (const baseName of config.baseNames) {
            const widget = node.widgets.find(w => w.name === `${baseName}${i}`);
            if (widget) {
                toggleWidget(node, widget, i <= count);
            }
        }
    }
}

app.registerExtension({
    name: "anima_xyplot.widgets",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        const config = NODE_CONFIG[nodeData.name];
        if (!config) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origOnNodeCreated?.apply(this, arguments);

            const node = this;
            requestAnimationFrame(() => {
                updateVisibility(node, config);

                const countWidget = node.widgets?.find(w => w.name === config.countWidget);
                if (countWidget) {
                    const origCallback = countWidget.callback;
                    countWidget.callback = function (...args) {
                        updateVisibility(node, config);
                        if (origCallback) origCallback.apply(this, args);
                    };

                    if (countWidget.inputEl) {
                        countWidget.inputEl.addEventListener("input", () => {
                            updateVisibility(node, config);
                        });
                        countWidget.inputEl.addEventListener("change", () => {
                            updateVisibility(node, config);
                        });
                    }

                    const origMouse = countWidget.mouse;
                    countWidget.mouse = function (...args) {
                        const result = origMouse?.apply(this, args);
                        requestAnimationFrame(() => updateVisibility(node, config));
                        return result;
                    };
                }
            });

            return result;
        };
    }
});
