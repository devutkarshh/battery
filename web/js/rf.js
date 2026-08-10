/*
 * rf.js - Browser Random Forest inference + feature engineering.
 *
 * The model (window.RUL_MODEL) was exported from
 *   outputs/battery_rul_rf_model.pkl  (see web/scripts/export_model.py)
 * and verified to reproduce sklearn to < 0.001 cycle (web/scripts/verify_model.py).
 *
 * The 14 model features, in exact order:
 *   chI, chV, chT, disI, disV, disT, BCt, SOH,
 *   capacity_fade_rate, SOH_rate, temp_rise_rate,
 *   voltage_drop, BCt_rolling_mean, SOH_rolling_mean
 */
(function () {
  "use strict";

  var MODEL = window.RUL_MODEL;
  var N_TRESS = MODEL ? MODEL.nTrees : 0;

  /* Predict RUL from a 14-element feature vector. */
  function predict(features) {
    if (!MODEL) throw new Error("Model not loaded");
    var trees = MODEL.trees;
    var n = trees.length;
    var sum = 0;
    for (var t = 0; t < n; t++) {
      var tr = trees[t];
      var node = 0;
      var left = tr.left, right = tr.right, feat = tr.feature, thr = tr.threshold;
      while (left[node] !== -1) {
        node = (features[feat[node]] <= thr[node]) ? left[node] : right[node];
      }
      sum += tr.value[node];
    }
    return sum / n;
  }

  /*
   * Build the engineered features for index `i` of a battery whose columns
   * are stored as per-column arrays (cycle, chI, chV, chT, disI, disV, disT,
   * BCt, SOH, RUL). Mirrors battery_rul_model.py feature engineering.
   * Returns the 14-length feature vector.
   */
  function buildFeatures(cols, i) {
    var mean = function (arr, fromIdx) {
      var lo = Math.max(0, fromIdx - 4);   // 5-point window
      var s = 0;
      for (var k = lo; k <= fromIdx; k++) s += arr[k];
      return s / (fromIdx - lo + 1);
    };

    var prev = Math.max(0, i - 1);
    var capacity_fade_rate = (i === 0) ? 0 : (cols.BCt[i] - cols.BCt[prev]);
    var SOH_rate = (i === 0) ? 0 : (cols.SOH[i] - cols.SOH[prev]);
    var temp_rise_rate = (i === 0) ? 0 : (cols.chT[i] - cols.chT[prev]);
    var voltage_drop = cols.chV[i] - cols.disV[i];
    var BCt_rolling_mean = mean(cols.BCt, i);
    var SOH_rolling_mean = mean(cols.SOH, i);

    return [
      cols.chI[i], cols.chV[i], cols.chT[i],
      cols.disI[i], cols.disV[i], cols.disT[i],
      cols.BCt[i], cols.SOH[i],
      capacity_fade_rate, SOH_rate, temp_rise_rate,
      voltage_drop, BCt_rolling_mean, SOH_rolling_mean
    ];
  }

  window.RUL = {
    predict: predict,
    buildFeatures: buildFeatures,
    features: MODEL ? MODEL.features : [],
    metadata: MODEL ? MODEL.metadata : null
  };
})();
