/* app.js - UI logic for the Battery RUL dashboard */
(function () {
  "use strict";

  var DATA = window.BATTERY_DATA;
  var COLORS = { B5: "#38bdf8", B6: "#34d399", B7: "#fb923c" };

  /* ---------- helpers ---------- */
  function $(s, c) { return (c || document).querySelector(s); }
  function $$(s, c) { return Array.prototype.slice.call((c || document).querySelectorAll(s)); }
  var fmt = function (n, d) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: d == null ? 1 : d }); };

  function batteryStats(b) {
    var cols = DATA[b];
    var initCap = Math.max.apply(null, cols.BCt);
    var minCap = Math.min.apply(null, cols.BCt);
    var cycles = cols.cycle.length;
    var fade = (initCap - minCap) / initCap * 100;
    var avgSOH = cols.SOH.reduce(function (a, x) { return a + x; }, 0) / cycles;
    return { cycles: cycles, initCap: initCap, minCap: minCap, fade: fade,
             avgSOH: avgSOH, finalSOH: cols.SOH[cycles - 1] };
  }

  /* ---------- hero animated background ---------- */
  function initHero() {
    var canvas = document.getElementById("hero-canvas");
    if (!canvas || !canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    var W, H, parts = [];
    function resize() {
      W = canvas.width = canvas.offsetWidth;
      H = canvas.height = canvas.offsetHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    var N = Math.min(90, Math.floor(window.innerWidth / 14));
    for (var i = 0; i < N; i++) {
      parts.push({
        x: Math.random() * W, y: Math.random() * H,
        r: 1 + Math.random() * 2.2,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        hue: Math.random() > 0.5 ? 190 : 155,
        pulse: Math.random() * Math.PI * 2
      });
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      for (var i = 0; i < parts.length; i++) {
        var p = parts[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
        if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
        p.pulse += 0.02;
        var a = 0.35 + 0.35 * Math.sin(p.pulse);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "hsla(" + p.hue + ", 95%, 65%, " + a.toFixed(3) + ")";
        ctx.fill();
        for (var j = i + 1; j < parts.length; j++) {
          var q = parts[j];
          var dx = p.x - q.x, dy = p.y - q.y;
          var d2 = dx * dx + dy * dy;
          if (d2 < 130 * 130) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
            var alpha = 0.12 * (1 - Math.sqrt(d2) / 130);
            ctx.strokeStyle = "hsla(190, 90%, 60%, " + alpha.toFixed(3) + ")";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
  }

  /* ---------- nav ---------- */
  function initNav() {
    var nav = $("#navbar");
    var onScroll = function () { nav.classList.toggle("scrolled", window.scrollY > 20); };
    window.addEventListener("scroll", onScroll);
    onScroll();
    var burger = $("#nav-toggle");
    var links = $("#nav-links");
    burger.addEventListener("click", function () { links.classList.toggle("open"); });
    $$("#nav-links a").forEach(function (a) {
      a.addEventListener("click", function () { links.classList.remove("open"); });
    });
  }

  /* ---------- reveal on scroll ---------- */
  function initReveal() {
    var els = $$(".reveal");
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (e) { e.classList.add("in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
      });
    }, { threshold: 0.12 });
    els.forEach(function (e) { io.observe(e); });
  }

  /* ---------- hero stat counters ---------- */
  function initCounters() {
    var set = window.PERF_STATS;
    if (!set) return;
    var done = false;
    var els = $$(".stat-num");
    var io = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (en) {
        if (en.isIntersecting && !done) {
          done = true; obs.disconnect();
          els.forEach(function (el) {
            var key = el.getAttribute("data-stat");
            var target = set[key];
            var suffix = el.getAttribute("data-suffix") || "";
            var t0 = null, dur = 1200;
            function step(ts) {
              if (!t0) t0 = ts;
              var p = Math.min(1, (ts - t0) / dur);
              var e2 = 1 - Math.pow(1 - p, 3);
              var val = target * e2;
              el.textContent = (Number.isInteger(target) ? Math.round(val) : val.toFixed(2)) + suffix;
              if (p < 1) requestAnimationFrame(step);
            }
            requestAnimationFrame(step);
          });
        }
      });
    }, { threshold: 0.4 });
    io.observe($(".hero-stats"));
  }

  /* ---------- feature importance chart ---------- */
  function initFeatureChart() {
    var fi = window.FEATURE_IMPORTANCE;
    var labels = Object.keys(fi).sort(function (a, b) { return fi[a] - fi[b]; });
    var values = labels.map(function (k) { return fi[k]; });
    var engineered = ["capacity_fade_rate", "SOH_rate", "temp_rise_rate", "voltage_drop", "BCt_rolling_mean", "SOH_rolling_mean"];
    var colors = labels.map(function (k) {
      return engineered.indexOf(k) >= 0 ? "rgba(52,211,153,0.85)" : "rgba(56,189,248,0.85)";
    });
    var ctx = $("#fi-chart");
    if (!ctx) return;
    new Chart(ctx, {
      type: "bar",
      data: { labels: labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6, borderSkipped: false }] },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: function (c) { return " importance: " + c.parsed.x.toFixed(3); } } } },
        scales: {
          x: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
          y: { grid: { display: false }, ticks: { color: "#cbd5e1", font: { size: 12 } } }
        }
      }
    });
  }

/* ---------- data explorer ---------- */
  var expCharts = [];
  function initExplorer() {
    var select = $("#explorer-metric");
    if (!select) return;
    renderExplorer(select.value);
    select.addEventListener("change", function () { renderExplorer(select.value); });
    $$(".bat-toggle").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var on = btn.classList.toggle("on");
        var b = btn.getAttribute("data-bat");
        btn.style.borderColor = on ? COLORS[b] : "";
        btn.style.color = on ? COLORS[b] : "";
        btn.style.background = on ? COLORS[b] + "22" : "";
        renderExplorer(select.value);
      });
    });
  }

  function activeBatteries() {
    return $$(".bat-toggle.on").map(function (b) { return b.getAttribute("data-bat"); });
  }

  function renderExplorer(metric) {
    var you = { SOH: "SOH (%)", BCt: "Capacity (Ah)", RUL: "Remaining useful life (cycles)" };
    expCharts.forEach(function (c) { c.destroy(); });
    expCharts = [];
    var canvas = $("#explorer-chart");
    if (!canvas) return;
    var ctx = canvas;
    var datasets = activeBatteries().map(function (b) {
      var cols = DATA[b];
      return {
        label: "Battery " + b,
        data: cols.cycle.map(function (_, i) { return { x: cols.cycle[i], y: cols[metric][i] }; }),
        borderColor: COLORS[b],
        backgroundColor: COLORS[b] + "22",
        fill: true, tension: 0.35, pointRadius: 0, pointHitRadius: 8, borderWidth: 2.5
      };
    });
    var chart = new Chart(ctx, {
      type: "line",
      data: { datasets: datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#cbd5e1", usePointStyle: true, pointStyle: "line" } },
          tooltip: { backgroundColor: "#0b1220", borderColor: "rgba(255,255,255,0.1)", borderWidth: 1,
                     titleColor: "#e2e8f0", bodyColor: "#cbd5e1" }
        },
        scales: {
          x: { type: "linear", title: { display: true, text: "Cycle", color: "#94a3b8" },
               grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
          y: { title: { display: true, text: you[metric], color: "#94a3b8" },
               grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
        }
      }
    });
    expCharts.push(chart);
  }

  /* ---------- RUL predictor ---------- */
  var predChart = null;
  function initPredictor() {
    var modeBtns = $$(".mode-btn");
    modeBtns.forEach(function (b) {
      b.addEventListener("click", function () {
        modeBtns.forEach(function (x) { x.classList.remove("active"); });
        b.classList.add("active");
        var m = b.getAttribute("data-mode");
        $("#dataset-fields").classList.toggle("hidden", m !== "dataset");
        $("#manual-fields").classList.toggle("hidden", m !== "manual");
      });
    });

    var batSel = $("#pred-battery");
    var cycSel = $("#pred-cycle");
    function fillCycle() {
      var n = DATA[batSel.value].cycle.length;
      cycSel.innerHTML = "";
      for (var i = 1; i <= n; i++) {
        var opt = document.createElement("option");
        opt.value = i; opt.textContent = "Cycle " + i;
        cycSel.appendChild(opt);
      }
      cycSel.value = Math.min(parseInt(cycSel.value, 10) || 1, n);
    }
    batSel.addEventListener("change", fillCycle);
    fillCycle();

    batSel.addEventListener("change", runDatasetPrediction);
    cycSel.addEventListener("change", runDatasetPrediction);
    $("#predict-btn").addEventListener("click", runPrediction);
    $("#predict-btn-manual").addEventListener("click", runManualPrediction);

    $$("input[type=range][data-link]").forEach(function (r) {
      var link = $("#" + r.getAttribute("data-link"));
      var upd = function () { link.value = r.value; link.textContent = r.value; };
      r.addEventListener("input", upd);
    });

    runDatasetPrediction();
  }

  function runPrediction() { runDatasetPrediction(); }

  function runDatasetPrediction() {
    var b = $("#pred-battery").value;
    var i = parseInt($("#pred-cycle").value, 10) - 1;
    var cols = DATA[b];
    var feats = window.RUL.buildFeatures(cols, i);
    var pred = window.RUL.predict(feats);
    showResult(pred, cols.RUL[i], b, i);
  }

  function runManualPrediction() {
    var g = function (id) { return parseFloat(document.getElementById(id).value); };
    var chV = g("m-chV"), disV = g("m-disV");
    var chI = g("m-chI"), disI = g("m-disI");
    var chT = g("m-chT"), disT = g("m-disT");
    var BCt = g("m-BCt"), SOH = g("m-SOH");

    /* engineered defaults for single-shot manual entry */
    var capacity_fade_rate = 0;
    var SOH_rate = 0;
    var temp_rise_rate = 0;
    var voltage_drop = chV - disV;
    var BCt_rolling_mean = BCt;
    var SOH_rolling_mean = SOH;

    var feats = [
      chI, chV, chT,
      disI, disV, disT,
      BCt, SOH,
      capacity_fade_rate, SOH_rate, temp_rise_rate,
      voltage_drop,
      BCt_rolling_mean, SOH_rolling_mean
    ];
    var pred = window.RUL.predict(feats);
    showResult(pred, null, null, null);
  }

  function showResult(pred, actual, bat, idx) {
    pred = Math.max(0, pred);
    var r = Math.round(pred);
    var gauge = $("#gauge");
    var pct = Math.min(1, pred / 250);
    gauge.style.background = "conic-gradient(#34d399, #38bdf8 " + (pct * 360) + "deg, rgba(255,255,255,0.08) " + (pct * 360) + "deg 360deg)";
    var num = $("#pred-number");
    num.textContent = r;
    num.style.color = "#34d399";

    var label, hue;
    if (pred >= 150) { label = "Healthy"; hue = "#34d399"; }
    else if (pred >= 80) { label = "Aging"; hue = "#fbbf24"; }
    else if (pred >= 30) { label = "Degraded"; hue = "#fb923c"; }
    else { label = "End of Life"; hue = "#f87171"; }
    var badge = $("#health-badge");
    badge.textContent = label;
    badge.style.background = hue + "22";
    badge.style.color = hue;
    badge.style.borderColor = hue;

    var ctxLine = $("#pred-context");
    if (actual != null) {
      var err = pred - actual;
      ctxLine.innerHTML = "Battery <b>" + bat + "</b>, cycle <b>" + (idx + 1) + "</b> &nbsp;·&nbsp; actual RUL = <b>" + Math.round(actual) + "</b> cycles &nbsp;·&nbsp; Δ = " + (err >= 0 ? "+" : "") + fmt(err, 1);
    } else {
      ctxLine.textContent = "Manual prediction from sensor readings";
    }

    if (bat && window.Chart && $("#spark")) {
      if (predChart) predChart.destroy();
      var cols = DATA[bat];
      predChart = new Chart($("#spark"), {
        type: "line",
        data: {
          labels: cols.cycle,
          datasets: [
            { label: "Actual RUL", data: cols.RUL, borderColor: "#64748b", pointRadius: 0, borderWidth: 1.5, fill: false },
            { label: "This prediction", data: cols.RUL.map(function (_, j) { return j === idx ? r : null; }),
              borderColor: "#34d399", pointRadius: 5, pointBackgroundColor: "#34d399", showLine: false }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#94a3b8", font: { size: 10 } } } },
          scales: {
            x: { title: { display: true, text: "Cycle", color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", font: { size: 9 } } },
            y: { title: { display: true, text: "RUL", color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", font: { size: 9 } } }
          }
        }
      });
    }
  }

  /* ---------- thermal gallery ---------- */
  var lightbox = null;
  function initThermal() {
    var tabs = $$(".tab-btn");
    function show(b) {
      tabs.forEach(function (t) { t.classList.toggle("active", t.getAttribute("data-bat") === b); });
      $$(".gallery-bat").forEach(function (g) { g.classList.toggle("hidden", g.getAttribute("data-bat") !== b); });
    }
    tabs.forEach(function (t) { t.addEventListener("click", function () { show(t.getAttribute("data-bat")); }); });

    lightbox = $("#lightbox");
    var boxImg = $("#lightbox-img");
    $$(".gallery-bat img").forEach(function (img) {
      img.addEventListener("click", function () {
        boxImg.src = img.src;
        $("#lightbox-cap").textContent = img.getAttribute("alt") || "";
        $("#lightbox-dl").href = img.src;
        lightbox.classList.add("open");
      });
    });
    $("#lightbox-close").addEventListener("click", closeLightbox);
    lightbox.addEventListener("click", function (e) { if (e.target === lightbox) closeLightbox(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeLightbox(); });

    /* inject per-figure download links */
    $$(".gallery-bat figure").forEach(function (fig) {
      var img = fig.querySelector("img");
      if (!img) return;
      var a = document.createElement("a");
      a.className = "fig-dl";
      a.href = img.src;
      a.target = "_blank";
      a.rel = "noopener";
      a.title = "Open / Download image";
      a.innerHTML = "&#11015;";
      fig.appendChild(a);
    });
  }
  function closeLightbox() { if (lightbox) lightbox.classList.remove("open"); }

  /* ---------- battery metric cards ---------- */
  function renderCards() {
    ["B5", "B6", "B7"].forEach(function (b) {
      var s = batteryStats(b);
      if ($("#card-" + b)) {
        $("#" + b + "-cycles").textContent = s.cycles;
        $("#" + b + "-fade").textContent = fmt(s.fade, 1) + "%";
        $("#" + b + "-soh").textContent = fmt(s.avgSOH, 1) + "%";
        $("#" + b + "-final").textContent = fmt(s.finalSOH, 1) + "%";
      }
    });
  }

  /* ---------- boot ---------- */
  function boot() {
    initHero();
    initNav();
    initReveal();
    initCounters();
    initFeatureChart();
    initExplorer();
    initPredictor();
    initThermal();
    renderCards();
  }

  if (document.readyState !== "loading") boot();
  else document.addEventListener("DOMContentLoaded", boot);
})();