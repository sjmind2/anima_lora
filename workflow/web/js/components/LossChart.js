(function () {
  window.VueComponents = window.VueComponents || {};

  var STAGE_COLORS = [
    "#3c78c8",
    "#27ae60",
    "#e67e22",
    "#9b59b6",
    "#e74c3c",
    "#1abc9c",
    "#f1c40f",
    "#e84393",
  ];

  var PADDING = { top: 20, right: 60, bottom: 30, left: 60 };

  function downsampleLTTB(data, threshold) {
    if (data.length <= threshold) return data;
    var sampled = [data[0]];
    var bucketSize = (data.length - 2) / (threshold - 2);
    var a = 0;
    for (var i = 0; i < threshold - 2; i++) {
      var avgStart = Math.floor((i + 0) * bucketSize) + 1;
      var avgEnd = Math.floor((i + 1) * bucketSize) + 1;
      var avgEndSafe = Math.min(avgEnd, data.length - 1);
      var avgX = 0;
      var avgY = 0;
      var count = 0;
      for (var j = avgStart; j < avgEndSafe; j++) {
        avgX += data[j].step;
        avgY += data[j].loss;
        count++;
      }
      if (count > 0) {
        avgX /= count;
        avgY /= count;
      }
      var rangeOff = Math.floor((i + 0) * bucketSize) + 1;
      var rangeTo = Math.floor((i + 1) * bucketSize) + 1;
      var maxDist = -1;
      var maxIdx = rangeOff;
      var ax = data[a].step;
      var ay = data[a].loss;
      var dx = avgX - ax;
      var dy = avgY - ay;
      var norm = Math.sqrt(dx * dx + dy * dy);
      for (var k = rangeOff; k < rangeTo && k < data.length - 1; k++) {
        var d = Math.abs(
          dx * (ay - data[k].loss) - (ax - data[k].step) * dy
        );
        if (norm > 0) d /= norm;
        if (d > maxDist) {
          maxDist = d;
          maxIdx = k;
        }
      }
      sampled.push(data[maxIdx]);
      a = maxIdx;
    }
    sampled.push(data[data.length - 1]);
    return sampled;
  }

  function downsampleMinMax(data, step) {
    var result = [];
    for (var i = 0; i < data.length; i += step) {
      var end = Math.min(i + step, data.length);
      var minVal = data[i];
      var maxVal = data[i];
      for (var j = i + 1; j < end; j++) {
        if (data[j].loss < minVal.loss) minVal = data[j];
        if (data[j].loss > maxVal.loss) maxVal = data[j];
      }
      if (minVal === maxVal) {
        result.push(minVal);
      } else if (minVal.step < maxVal.step) {
        result.push(minVal);
        result.push(maxVal);
      } else {
        result.push(maxVal);
        result.push(minVal);
      }
    }
    return result;
  }

  function downsample(data) {
    if (data.length <= 500) return data;
    if (data.length <= 2000) return downsampleMinMax(data, 2);
    return downsampleLTTB(data, 500);
  }

  function computeEMA(data, alpha) {
    if (data.length === 0) return [];
    var ema = [{ step: data[0].step, loss: data[0].loss }];
    var val = data[0].loss;
    for (var i = 1; i < data.length; i++) {
      val = alpha * data[i].loss + (1 - alpha) * val;
      ema.push({ step: data[i].step, loss: val });
    }
    return ema;
  }

  function niceNum(range, round) {
    var exp = Math.floor(Math.log10(Math.max(range, 1e-10)));
    var frac = range / Math.pow(10, exp);
    var nice;
    if (round) {
      if (frac < 1.5) nice = 1;
      else if (frac < 3) nice = 2;
      else if (frac < 7) nice = 5;
      else nice = 10;
    } else {
      if (frac <= 1) nice = 1;
      else if (frac <= 2) nice = 2;
      else if (frac <= 5) nice = 5;
      else nice = 10;
    }
    return nice * Math.pow(10, exp);
  }

  function niceScale(min, max, ticks) {
    var range = niceNum(max - min, false);
    var spacing = niceNum(range / (ticks - 1), true);
    var nMin = Math.floor(min / spacing) * spacing;
    var nMax = Math.ceil(max / spacing) * spacing;
    return { min: nMin, max: nMax, spacing: spacing };
  }

  window.VueComponents.LossChart = {
    props: {
      lossData: {
        type: Array,
        default: function () { return []; },
      },
    },
    data: function () {
      return {
        svgWidth: 600,
        svgHeight: 200,
        mouseX: -1,
        mouseY: -1,
        showCrosshair: false,
      };
    },
    computed: {
      stageIds: function () {
        var seen = {};
        var result = [];
        this.lossData.forEach(function (d) {
          if (d.stage_id && !seen[d.stage_id]) {
            seen[d.stage_id] = true;
            result.push(d.stage_id);
          }
        });
        return result;
      },
      groupedData: function () {
        var groups = {};
        var self = this;
        this.lossData.forEach(function (d) {
          var key = d.stage_id || "__default";
          if (!groups[key]) groups[key] = [];
          groups[key].push({ step: d.step, loss: d.loss });
        });
        var result = {};
        Object.keys(groups).forEach(function (key) {
          var sorted = groups[key].slice().sort(function (a, b) {
            return a.step - b.step;
          });
          result[key] = {
            raw: downsample(sorted),
            ema: computeEMA(downsample(sorted), 0.05),
          };
        });
        return result;
      },
      chartBounds: function () {
        var minStep = Infinity;
        var maxStep = -Infinity;
        var minLoss = Infinity;
        var maxLoss = -Infinity;
        var gd = this.groupedData;
        Object.keys(gd).forEach(function (key) {
          gd[key].raw.forEach(function (d) {
            if (d.step < minStep) minStep = d.step;
            if (d.step > maxStep) maxStep = d.step;
            if (d.loss < minLoss) minLoss = d.loss;
            if (d.loss > maxLoss) maxLoss = d.loss;
          });
        });
        if (!isFinite(minStep)) {
          return { minStep: 0, maxStep: 100, minLoss: 0, maxLoss: 1 };
        }
        var lossPad = (maxLoss - minLoss) * 0.1 || 0.1;
        return {
          minStep: minStep,
          maxStep: maxStep,
          minLoss: Math.max(0, minLoss - lossPad),
          maxLoss: maxLoss + lossPad,
        };
      },
      xScale: function () {
        var b = this.chartBounds;
        var w = this.svgWidth - PADDING.left - PADDING.right;
        var range = b.maxStep - b.minStep || 1;
        return function (step) {
          return PADDING.left + ((step - b.minStep) / range) * w;
        };
      },
      yScale: function () {
        var b = this.chartBounds;
        var h = this.svgHeight - PADDING.top - PADDING.bottom;
        var range = b.maxLoss - b.minLoss || 1;
        return function (loss) {
          return PADDING.top + h - ((loss - b.minLoss) / range) * h;
        };
      },
      yTicks: function () {
        var b = this.chartBounds;
        var scale = niceScale(b.minLoss, b.maxLoss, 5);
        var ticks = [];
        for (var v = scale.min; v <= scale.max + scale.spacing * 0.5; v += scale.spacing) {
          ticks.push(v);
        }
        return ticks;
      },
      xTicks: function () {
        var b = this.chartBounds;
        var scale = niceScale(b.minStep, b.maxStep, 6);
        var ticks = [];
        for (var v = scale.min; v <= scale.max + scale.spacing * 0.5; v += scale.spacing) {
          ticks.push(v);
        }
        return ticks;
      },
      stageBoundaries: function () {
        if (this.stageIds.length <= 1) return [];
        var boundaries = [];
        var self = this;
        this.stageIds.forEach(function (sid, idx) {
          if (idx > 0) {
            var data = self.lossData.filter(function (d) {
              return (d.stage_id || "__default") === sid;
            });
            if (data.length > 0) {
              boundaries.push(data[0].step);
            }
          }
        });
        return boundaries;
      },
      paths: function () {
        var result = [];
        var self = this;
        var xs = this.xScale;
        var ys = this.yScale;
        var idx = 0;
        Object.keys(this.groupedData).forEach(function (key) {
          var color = STAGE_COLORS[idx % STAGE_COLORS.length];
          var group = self.groupedData[key];
          var rawD = group.raw.map(function (p) {
            return xs(p.step).toFixed(1) + "," + ys(p.loss).toFixed(1);
          }).join(" ");
          var emaD = group.ema.map(function (p) {
            return xs(p.step).toFixed(1) + "," + ys(p.loss).toFixed(1);
          }).join(" ");
          result.push({
            key: key,
            color: color,
            rawPath: rawD ? "M" + rawD : "",
            emaPath: emaD ? "M" + emaD : "",
          });
          idx++;
        });
        return result;
      },
      crosshairData: function () {
        if (!this.showCrosshair || this.mouseX < 0) return null;
        var xs = this.xScale;
        var ys = this.yScale;
        var b = this.chartBounds;
        var w = this.svgWidth - PADDING.left - PADDING.right;
        var stepVal = b.minStep + ((this.mouseX - PADDING.left) / w) * (b.maxStep - b.minStep);
        var closest = null;
        var closestDist = Infinity;
        var self = this;
        this.lossData.forEach(function (d) {
          var dist = Math.abs(d.step - stepVal);
          if (dist < closestDist) {
            closestDist = dist;
            closest = d;
          }
        });
        if (!closest) return null;
        return {
          step: closest.step,
          loss: closest.loss,
          lr: closest.lr,
          stageId: closest.stage_id,
          cx: xs(closest.step),
          cy: ys(closest.loss),
        };
      },
    },
    mounted: function () {
      this.updateSize();
      var self = this;
      if (window.ResizeObserver && this.$refs.chartContainer) {
        this._resizeObserver = new ResizeObserver(function () {
          self.updateSize();
        });
        this._resizeObserver.observe(this.$refs.chartContainer);
      }
    },
    beforeUnmount: function () {
      if (this._resizeObserver) {
        this._resizeObserver.disconnect();
      }
    },
    methods: {
      updateSize: function () {
        var el = this.$refs.chartContainer;
        if (el) {
          this.svgWidth = el.clientWidth || 600;
          this.svgHeight = el.clientHeight || 200;
        }
      },
      onMouseMove: function (e) {
        var rect = this.$refs.svgEl.getBoundingClientRect();
        this.mouseX = e.clientX - rect.left;
        this.mouseY = e.clientY - rect.top;
        this.showCrosshair = true;
      },
      onMouseLeave: function () {
        this.showCrosshair = false;
        this.mouseX = -1;
      },
    },
    template: [
      '<div class="loss-chart-container" ref="chartContainer">',
      '  <svg :width="svgWidth" :height="svgHeight" ref="svgEl"',
      '    @mousemove="onMouseMove"',
      '    @mouseleave="onMouseLeave"',
      '    class="loss-chart-svg">',
      '    <g v-for="p in paths" :key="p.key">',
      '      <polyline :points="p.rawPath" fill="none" :stroke="p.color" stroke-width="1" opacity="0.3" />',
      '      <polyline :points="p.emaPath" fill="none" :stroke="p.color" stroke-width="2" opacity="0.9" />',
      '    </g>',
      '    <line v-for="b in stageBoundaries" :key="b"',
      '      :x1="xScale(b)" :y1="PADDING.top"',
      '      :x2="xScale(b)" :y2="svgHeight - PADDING.bottom"',
      '      stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="4,4" opacity="0.5" />',
      '    <line v-for="t in yTicks" :key="\'y\'+t"',
      '      :x1="PADDING.left" :y1="yScale(t)"',
      '      :x2="svgWidth - PADDING.right" :y2="yScale(t)"',
      '      stroke="var(--border-light)" stroke-width="0.5" />',
      '    <text v-for="t in yTicks" :key="\'yt\'+t"',
      '      :x="PADDING.left - 8" :y="yScale(t) + 4"',
      '      text-anchor="end" fill="var(--text-dim)" font-size="11"',
      '      font-family="var(--font)">{{ t.toFixed(4) }}</text>',
      '    <text v-for="t in xTicks" :key="\'xt\'+t"',
      '      :x="xScale(t)" :y="svgHeight - PADDING.bottom + 18"',
      '      text-anchor="middle" fill="var(--text-dim)" font-size="11"',
      '      font-family="var(--font)">{{ t }}</text>',
      '    <g v-if="crosshairData">',
      '      <line :x1="crosshairData.cx" :y1="PADDING.top"',
      '        :x2="crosshairData.cx" :y2="svgHeight - PADDING.bottom"',
      '        stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="2,2" />',
      '      <circle :cx="crosshairData.cx" :cy="crosshairData.cy" r="4"',
      '        fill="var(--accent)" stroke="var(--bg-window)" stroke-width="2" />',
      '      <foreignObject :x="crosshairData.cx + 10" :y="crosshairData.cy - 30" width="180" height="50">',
      '        <div class="lc-tooltip" xmlns="http://www.w3.org/1999/xhtml">',
      '          <div>Step: {{ crosshairData.step }} | Loss: {{ crosshairData.loss.toFixed(6) }}</div>',
      '          <div v-if="crosshairData.lr">LR: {{ crosshairData.lr }}</div>',
      '          <div v-if="crosshairData.stageId" class="lc-tooltip-stage">{{ crosshairData.stageId }}</div>',
      '        </div>',
      '      </foreignObject>',
      '    </g>',
      '  </svg>',
      '  <div v-if="lossData.length === 0" class="lc-empty">暂无训练数据</div>',
      '</div>',
    ].join("\n"),
  };
})();
