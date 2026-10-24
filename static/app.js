/* anomality frontend: watchlist scanner + per-asset analysis dashboard. */
(() => {
  "use strict";

  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  const state = {
    lists: { stocks: [], crypto: [] },
    custom: store.get("anomality.custom", []),
    group: store.get("anomality.group", "all"),
    scan: {},          // symbol -> summary
    selected: store.get("anomality.selected", null),
    detail: null,      // last /api/analyze response
    priceRange: "1Y",
    tableFilter: "all",
    tableSort: { key: "date", dir: -1 },
    scanToken: 0,
    detailToken: 0,
  };

  const VERDICT = {
    HIGH: { icon: "▲", label: "High", text: "Unusual movement likely" },
    ELEVATED: { icon: "◆", label: "Elevated", text: "Conditions are elevated" },
    NORMAL: { icon: "●", label: "Normal", text: "No unusual movement expected" },
  };

  // ---------- helpers ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
  const signed = (v, d = 2) => (v == null ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(d)}`);
  function money(v, cur = "USD") {
    if (v == null) return "–";
    const digits = v >= 1000 ? 2 : v >= 1 ? 2 : v >= 0.01 ? 4 : 6;
    const sym = cur === "USD" ? "$" : "";
    return sym + v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }) + (sym ? "" : ` ${cur}`);
  }
  const badge = (v) => v ? `<span class="badge ${v}">${VERDICT[v].icon} ${VERDICT[v].label}</span>` : `<span class="badge ERR">n/a</span>`;

  function params() {
    return new URLSearchParams({
      period: $("#period").value,
      threshold: $("#threshold").value,
      horizon: $("#horizon").value,
      model: $("#model").value,
      depth: $("#depth").value,
    });
  }

  async function api(path) {
    const res = await fetch(path);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
    return body;
  }

  function allSymbols() {
    return [...new Set([...state.lists.stocks, ...state.lists.crypto, ...state.custom])];
  }

  // ---------- watchlist ----------
  function groupSymbols() {
    if (state.group === "stocks") return state.lists.stocks;
    if (state.group === "crypto") return state.lists.crypto;
    if (state.group === "custom") return state.custom;
    return allSymbols();
  }

  function sparkline(values, up) {
    if (!values || values.length < 2) return "";
    const min = Math.min(...values), max = Math.max(...values), w = 64, h = 26;
    const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - 2 - ((v - min) / (max - min || 1)) * (h - 4)}`).join(" ");
    return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="${up ? "var(--good)" : "var(--down)"}" stroke-width="1.5" stroke-linejoin="round"/></svg>`;
  }

  function renderWatchlist() {
    const sortKey = $("#sort").value;
    const syms = [...groupSymbols()];
    const val = (s) => state.scan[s] || {};
    const sorters = {
      risk: (a, b) => (val(b).risk_score ?? -1) - (val(a).risk_score ?? -1),
      anomaly: (a, b) => (val(b).anomaly_score ?? -1) - (val(a).anomaly_score ?? -1),
      change: (a, b) => Math.abs(val(b).change_pct ?? 0) - Math.abs(val(a).change_pct ?? 0),
      symbol: (a, b) => a.localeCompare(b),
    };
    syms.sort(sorters[sortKey]);

    const ul = $("#watchlist");
    if (!syms.length) {
      ul.innerHTML = `<li style="cursor:default;display:block" class="muted small">${state.group === "custom" ? "Use the search box to add your own tickers." : "No symbols."}</li>`;
      return;
    }
    ul.innerHTML = syms.map((s) => {
      const r = state.scan[s];
      const removable = state.custom.includes(s) && !state.lists.stocks.includes(s) && !state.lists.crypto.includes(s);
      const rm = removable ? `<button class="wl-remove" data-remove="${esc(s)}" title="Remove">×</button>` : "";
      if (!r) {
        return `<li data-sym="${esc(s)}" class="${s === state.selected ? "selected" : ""}"><div><div class="wl-sym">${esc(s)}</div><div class="skeleton" style="width:80px;margin-top:4px"></div></div><div class="skeleton"></div><div class="skeleton" style="width:60px"></div>${rm}</li>`;
      }
      if (r.error) {
        return `<li data-sym="${esc(s)}" class="${s === state.selected ? "selected" : ""}"><div><div class="wl-sym">${esc(s)}</div><div class="wl-name" title="${esc(r.error)}">${esc(r.error)}</div></div><div></div><div class="wl-right">${badge(null)}</div>${rm}</li>`;
      }
      const up = r.change_pct >= 0;
      return `<li data-sym="${esc(s)}" class="${s === state.selected ? "selected" : ""}">
        <div><div class="wl-sym">${esc(s)}${r.anomaly_today ? ` <span title="Anomaly today: ${esc(r.anomaly_today)}" style="color:var(--other)">✦</span>` : ""}</div>
          <div class="wl-name" title="${esc(r.name)}">${esc(r.name)}</div></div>
        ${sparkline(r.spark, up)}
        <div class="wl-right"><div>${money(r.price, r.currency)}</div>
          <div class="${up ? "up" : "down"}">${signed(r.change_pct)}%</div>
          <div style="margin-top:3px" title="Risk score ${r.risk_score}">${badge(r.verdict)}</div></div>
        ${rm}</li>`;
    }).join("");
  }

  async function scan(symbols = allSymbols()) {
    if (!symbols.length) return;
    const token = ++state.scanToken;
    symbols.forEach((s) => delete state.scan[s]);
    renderWatchlist();
    $("#scan-status").textContent = `Scanning ${symbols.length} assets…`;
    const t0 = performance.now();
    try {
      // Scan in small batches so the list fills in progressively.
      const batches = [];
      for (let i = 0; i < symbols.length; i += 6) batches.push(symbols.slice(i, i + 6));
      await Promise.all(batches.map(async (batch) => {
        const p = params(); p.set("symbols", batch.join(","));
        const { results } = await api(`/api/scan?${p}`);
        if (token !== state.scanToken) return;
        results.forEach((r) => { state.scan[r.symbol.toUpperCase()] = r; });
        batch.forEach((s) => { if (!state.scan[s]) state.scan[s] = results.find((r) => r.symbol.toUpperCase() === s) || { symbol: s, error: "No data" }; });
        renderWatchlist();
      }));
      if (token !== state.scanToken) return;
      const flagged = symbols.filter((s) => state.scan[s]?.verdict === "HIGH").length;
      $("#scan-status").textContent = `Scanned ${symbols.length} assets in ${((performance.now() - t0) / 1000).toFixed(1)}s · ${flagged} flagged high risk.`;
    } catch (e) {
      $("#scan-status").textContent = `Scan failed: ${e.message}`;
    }
  }

  // ---------- detail ----------
  async function select(sym) {
    state.selected = sym;
    store.set("anomality.selected", sym);
    renderWatchlist();
    const token = ++state.detailToken;
    const main = $("#detail");
    if (!state.detail || state.detail.summary.symbol.toUpperCase() !== sym) {
      main.innerHTML = `<div class="loading"><div class="spinner"></div>Analyzing ${esc(sym)}: fetching prices, detecting anomalies, training the model…</div>`;
    } else {
      main.style.opacity = 0.55;
    }
    try {
      const d = await api(`/api/analyze/${encodeURIComponent(sym)}?${params()}`);
      if (token !== state.detailToken) return;
      state.detail = d;
      // Keep the sidebar in sync with the freshly computed summary.
      state.scan[sym] = { ...d.summary };
      renderWatchlist();
      renderDetail(d);
    } catch (e) {
      if (token !== state.detailToken) return;
      main.innerHTML = `<div class="error">Could not analyze ${esc(sym)}: ${esc(e.message)}</div>`;
    } finally {
      main.style.opacity = 1;
    }
  }

  function gauge(score, verdict) {
    const r = 46, cx = 56, cy = 60, len = Math.PI * r;
    const color = verdict === "HIGH" ? "var(--critical)" : verdict === "ELEVATED" ? "var(--warning)" : "var(--good)";
    const filled = len * Math.max(0, Math.min(1, score / 100));
    // Tick marks at the 50 and 65 decision boundaries.
    const tick = (v) => {
      const a = Math.PI * (1 - v / 100);
      return `<line x1="${cx + (r - 9) * Math.cos(a)}" y1="${cy - (r - 9) * Math.sin(a)}" x2="${cx + (r + 9) * Math.cos(a)}" y2="${cy - (r + 9) * Math.sin(a)}" stroke="var(--muted)" stroke-width="1.5"/>`;
    };
    return `<svg class="gauge" viewBox="0 0 112 112" role="img" aria-label="Risk score ${score} of 100">
      <path d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}" fill="none" stroke="var(--surface-2)" stroke-width="12" stroke-linecap="round"/>
      <path d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}" fill="none" stroke="${color}" stroke-width="12" stroke-linecap="round" stroke-dasharray="${filled} ${len}"/>
      ${tick(50)}${tick(65)}
      <text x="${cx}" y="${cy - 4}" text-anchor="middle" font-size="26" font-weight="700" fill="var(--ink)">${Math.round(score)}</text>
      <text x="${cx}" y="${cy + 14}" text-anchor="middle" font-size="10" fill="var(--muted)">risk / 100</text>
      <text x="${cx}" y="${cy + 40}" text-anchor="middle" font-size="11" font-weight="600" fill="var(--ink-2)">${VERDICT[verdict].icon} ${VERDICT[verdict].label}</text>
    </svg>`;
  }

  function renderDetail(d) {
    const s = d.summary, m = d.model, mt = m.metrics, set = d.settings;
    const up = s.change_pct >= 0;
    const lift = m.leaf.lift;
    const modelName = m.model === "forest" ? "random forest" : "decision tree";
    const reliability = mt.auc == null ? "unknown" : mt.auc >= 0.65 ? "good" : mt.auc >= 0.57 ? "modest" : "weak";

    $("#detail").innerHTML = `
      <div class="asset-head">
        <h1>${esc(s.symbol)}</h1><span class="name">${esc(s.name)}</span>
        <span class="price num">${money(s.price, s.currency)} <span class="${up ? "up" : "down"}" style="font-size:15px">${signed(s.change_pct)}%</span></span>
      </div>
      <div class="asset-sub">${esc(s.exchange)} · ${esc(s.type)} · last close ${esc(s.last_date)}${s.session_in_progress ? " · today's session is still open and not included yet" : ""}</div>
      ${s.synthetic ? `<div class="banner">⚠ Live data could not be fetched, so this shows <b>synthetic demo data</b>. Check your internet connection.</div>` : ""}

      <div class="grid top">
        <div class="card verdict">
          <h2>Decision model verdict</h2>
          <div class="score-row">
            ${gauge(m.risk_score, m.verdict)}
            <div class="big">Next ${set.horizon} day${set.horizon > 1 ? "s" : ""}<strong>${VERDICT[m.verdict].text}</strong>
              <span class="muted small">Unusual = a daily move beyond ${set.threshold}σ</span></div>
          </div>
          <p>Historically, <b>${m.leaf.followed_by_shock} of ${m.leaf.days}</b> days with conditions like today (${pct(m.leaf.rate)}) were followed by an unusual move within ${set.horizon} days, vs. <b>${pct(mt.base_rate)}</b> on an average day${lift != null ? ` (<b>${lift.toFixed(2)}×</b>)` : ""}.</p>
          <p class="small muted">Score from the ${modelName}. Its track record on unseen data is <b>${reliability}</b> (AUC ${mt.auc ?? "–"}). See Model performance below.</p>
        </div>
        <div class="card">
          <h2>Today at a glance</h2>
          <p class="sub">Anomaly detection on the latest session</p>
          <div class="tiles">
            <div class="tile"><div class="label">Anomaly score</div><div class="value">${Math.round(s.anomaly_score)}</div><div class="note">${s.anomaly_today ? esc(s.anomaly_today) : "Nothing unusual today"}</div></div>
            <div class="tile"><div class="label">Anomalies, last 30 days</div><div class="value">${s.anomalies_30d}</div><div class="note">${d.anomalies.length} over the full history</div></div>
            <div class="tile"><div class="label">Move today</div><div class="value">${signed(m.current_features.ret_z, 1)}σ</div><div class="note">vs. the last 20 days' volatility</div></div>
            <div class="tile"><div class="label">Volume today</div><div class="value">${signed(m.current_features.vol_z, 1)}σ</div><div class="note">vs. the 20-day average volume</div></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div><h2>Price and detected anomalies</h2><p class="sub">Markers show days flagged by z-score rules or the Isolation Forest. Hover for details, drag to zoom.</p></div>
          <div class="seg" id="price-range">${["3M", "6M", "1Y", "All"].map((r) => `<button data-r="${r}" class="${r === state.priceRange ? "active" : ""}">${r}</button>`).join("")}</div>
        </div>
        <div class="legend"><span><i style="background:var(--ink-2)"></i>Close</span><span><i class="tri-up"></i>Spike up</span><span><i class="tri-down"></i>Crash</span><span><i class="dot"></i>Volume / gap / multi-factor</span><span><i style="background:var(--axis)"></i>Volume</span></div>
        <div id="price-chart" class="chart" style="height:420px"></div>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>Model risk score over time</h2>
        <p class="sub">Out-of-sample predictions since ${esc(mt.test_start)} (the model never saw these days during training). Red ticks mark days with an actual unusual move. A useful model rises <i>before</i> clusters of ticks.</p>
        <div id="risk-chart" class="chart" style="height:240px"></div>
      </div>

      <div class="grid two">
        <div class="card">
          <h2>Why: today's decision path</h2>
          <p class="sub">The rules the decision tree applied to today's market conditions</p>
          <ol class="path">${m.path.map((p) => `<li><span class="cond">${esc(p.label)} is <b>${p.value}</b> ${p.op === "<=" ? "≤" : ">"} ${p.threshold}</span><span class="rate">${p.node_rate != null ? `${pct(p.node_rate, 0)} shock rate` : ""}</span></li>`).join("") || `<li><span class="cond">The tree has no splits.</span></li>`}</ol>
          <div class="path"><div class="final">→ Lands in a group of <b>${m.leaf.days}</b> historical days where <b>${pct(m.leaf.rate)}</b> saw an unusual move within ${set.horizon} days.</div></div>
        </div>
        <div class="card">
          <h2>What drives the model</h2>
          <p class="sub">Feature importance in the ${modelName}</p>
          <div id="imp-chart" class="chart" style="height:${Math.max(180, m.importance.length * 30 + 40)}px"></div>
        </div>
      </div>

      <div class="grid two">
        <div class="card">
          <div class="card-head"><div><h2>Decision tree</h2><p class="sub">Each node: share of days followed by an unusual move. Today's path is highlighted. Click a node to collapse it.</p></div></div>
          <div class="tree" id="tree"></div>
        </div>
        <div class="card">
          <div class="card-head">
            <div><h2>Anomaly log</h2><p class="sub">Click a row to zoom the chart to that date</p></div>
            <div class="seg" id="table-filter">${[["all", "All"], ["price", "Price"], ["volume", "Volume"], ["outlier", "Outlier"]].map(([k, l]) => `<button data-f="${k}" class="${k === state.tableFilter ? "active" : ""}">${l}</button>`).join("")}</div>
          </div>
          <div class="table-wrap" id="anom-table"></div>
        </div>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>Model performance</h2>
        <p class="sub">Trained on the first ${mt.train_days} days, tested on the most recent ${mt.test_days} days it never saw</p>
        <div class="metrics">
          <div class="metric"><div class="v">${mt.auc ?? "–"}</div><div class="l">AUC (0.5 = coin flip, 1 = perfect)</div></div>
          <div class="metric"><div class="v">${pct(mt.precision, 0)}</div><div class="l">Precision: alerts that came true (test base rate ${pct(mt.test_base_rate, 0)})</div></div>
          <div class="metric"><div class="v">${pct(mt.recall, 0)}</div><div class="l">Recall: unusual moves that were flagged in advance</div></div>
        </div>
        <p class="explain"><b>How it works.</b> Each day is described by ${Object.keys(m.current_features).length} indicators that use only past data: return and volume z-scores, volatility regime, gaps, momentum, RSI, Bollinger %B and recent shock counts. The target is whether a move larger than ${set.threshold}σ happens within the next ${set.horizon} days. A ${modelName} learns which combinations of indicators came before such moves; the score is its probability, balanced so that 50 means "as likely as average".</p>
        <p class="explain"><b>Read with care.</b> Markets are noisy, so even good scores are probabilities, not certainties. Volatility tends to cluster, which is mostly what the model picks up. Compare precision with the base rate: precision above the base rate means the alerts add information.</p>
      </div>`;

    drawPrice(d);
    drawRisk(d);
    drawImportance(d);
    renderTree(m.tree);
    renderTable();
    bindDetailEvents();
  }

  function plotBase() {
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 11, color: css("--muted") },
      margin: { l: 56, r: 12, t: 8, b: 28 },
      hovermode: "x unified",
      hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--axis"), font: { color: css("--ink"), size: 12 } },
      showlegend: false,
      dragmode: "zoom",
    };
  }
  const axis = (extra = {}) => ({ gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted") }, ...extra });
  const plotCfg = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"] };

  function rangeStart(dates, r) {
    if (r === "All") return dates[0];
    const end = new Date(dates[dates.length - 1]);
    const months = { "3M": 3, "6M": 6, "1Y": 12 }[r];
    end.setMonth(end.getMonth() - months);
    return end.toISOString().slice(0, 10);
  }

  function drawPrice(d, focusDate) {
    const s = d.series, dates = s.date;
    const idx = Object.fromEntries(dates.map((x, i) => [x, i]));
    const groups = { up: [], down: [], other: [] };
    d.anomalies.forEach((a) => {
      const g = /spike/i.test(a.type) ? "up" : /crash/i.test(a.type) ? "down" : "other";
      groups[g].push(a);
    });
    const marker = (list, color, symbol, name) => ({
      type: "scatter", mode: "markers", name, x: list.map((a) => a.date), y: list.map((a) => a.close),
      marker: { color, symbol, size: 10, line: { color: css("--surface"), width: 1.5 } },
      text: list.map((a) => `<b>${a.type}</b><br>Move ${signed(a.ret_pct)}% (${signed(a.ret_z, 1)}σ)<br>Volume ${a.vol_z == null ? "–" : signed(a.vol_z, 1) + "σ"}<br>Score ${Math.round(a.score)}`),
      hovertemplate: "%{text}<extra></extra>", xaxis: "x", yaxis: "y",
    });
    const anomSet = new Set(d.anomalies.map((a) => a.date));
    const traces = [
      { type: "scatter", mode: "lines", name: "Close", x: dates, y: s.close, line: { color: css("--ink-2"), width: 1.5 },
        hovertemplate: "Close %{y:,.4~f}<extra></extra>", xaxis: "x", yaxis: "y" },
      marker(groups.up, css("--up"), "triangle-up", "Spike up"),
      marker(groups.down, css("--down"), "triangle-down", "Crash"),
      marker(groups.other, css("--other"), "circle", "Other anomaly"),
      { type: "bar", name: "Volume", x: dates, y: s.volume, xaxis: "x", yaxis: "y2",
        marker: { color: dates.map((x) => (anomSet.has(x) ? css("--other") : css("--axis"))) },
        hovertemplate: "Volume %{y:,.3s}<extra></extra>" },
    ];
    let x0 = rangeStart(dates, state.priceRange), x1 = dates[dates.length - 1];
    if (focusDate && idx[focusDate] != null) {
      const i = idx[focusDate];
      x0 = dates[Math.max(0, i - 45)]; x1 = dates[Math.min(dates.length - 1, i + 45)];
    }
    const visible = dates.map((x, i) => (x >= x0 && x <= x1 ? i : -1)).filter((i) => i >= 0);
    const lo = Math.min(...visible.map((i) => s.close[i])), hi = Math.max(...visible.map((i) => s.close[i]));
    const pad = (hi - lo) * 0.08 || hi * 0.05;
    const shapes = focusDate ? [{ type: "line", xref: "x", yref: "paper", x0: focusDate, x1: focusDate, y0: 0, y1: 1, line: { color: css("--accent"), width: 1, dash: "dot" } }] : [];
    Plotly.react("price-chart", traces, {
      ...plotBase(),
      xaxis: axis({ range: [x0, x1], rangeslider: { visible: false }, type: "date", anchor: "y2" }),
      yaxis: axis({ domain: [0.28, 1], range: [lo - pad, hi + pad], tickformat: ",.4~f" }),
      yaxis2: axis({ domain: [0, 0.2], tickformat: ".2s", showgrid: false }),
      shapes,
      bargap: 0.15,
    }, plotCfg);
  }

  function drawRisk(d) {
    const rh = d.model.risk_history, thr = d.settings.threshold;
    const oos = rh.filter((r) => r.oos), live = rh.filter((r) => !r.oos);
    if (oos.length && live.length) live.unshift(oos[oos.length - 1]);
    const start = rh.length ? rh[0].date : null;
    const shocks = d.anomalies.filter((a) => Math.abs(a.ret_z) > thr && (!start || a.date >= start));
    const traces = [
      { type: "scatter", mode: "lines", name: "Risk (out-of-sample)", x: oos.map((r) => r.date), y: oos.map((r) => r.risk),
        line: { color: css("--accent"), width: 2 }, hovertemplate: "Risk %{y:.0f}<extra></extra>" },
      { type: "scatter", mode: "lines+markers", name: "Risk (latest, outcome pending)", x: live.map((r) => r.date), y: live.map((r) => r.risk),
        line: { color: css("--accent"), width: 2, dash: "dot" }, marker: { size: 8, color: css("--accent"), line: { color: css("--surface"), width: 2 } },
        hovertemplate: "Risk %{y:.0f} (pending)<extra></extra>" },
      { type: "scatter", mode: "markers", name: "Unusual move", x: shocks.map((a) => a.date), y: shocks.map(() => 3),
        marker: { symbol: "line-ns", size: 12, line: { color: css("--down"), width: 2 } },
        text: shocks.map((a) => `${a.type} ${signed(a.ret_pct)}%`), hovertemplate: "%{text}<extra></extra>" },
    ];
    const band = (y0, y1, color) => ({ type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0, y1, fillcolor: color, opacity: 0.07, line: { width: 0 }, layer: "below" });
    Plotly.react("risk-chart", traces, {
      ...plotBase(),
      xaxis: axis({ type: "date" }),
      yaxis: axis({ range: [0, 100], tickvals: [0, 50, 65, 100], title: { text: "Risk", font: { size: 11 } } }),
      shapes: [band(50, 65, css("--warning")), band(65, 100, css("--critical"))],
      annotations: [
        { xref: "paper", yref: "y", x: 1, y: 65, text: "High", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: css("--muted") } },
        { xref: "paper", yref: "y", x: 1, y: 50, text: "Elevated", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: css("--muted") } },
      ],
    }, plotCfg);
  }

  function drawImportance(d) {
    const imp = [...d.model.importance].reverse();
    Plotly.react("imp-chart", [{
      type: "bar", orientation: "h", x: imp.map((r) => r.value), y: imp.map((r) => r.label),
      marker: { color: css("--accent") }, text: imp.map((r) => `${(r.value * 100).toFixed(0)}%`), textposition: "outside",
      textfont: { color: css("--ink-2") }, cliponaxis: false,
      hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
    }], {
      ...plotBase(), hovermode: "closest", margin: { l: 190, r: 40, t: 4, b: 24 },
      xaxis: axis({ tickformat: ".0%", showgrid: true }), yaxis: axis({ gridcolor: "rgba(0,0,0,0)", tickfont: { color: css("--ink-2") } }),
      bargap: 0.35,
    }, { ...plotCfg, displayModeBar: false });
  }

  function renderTree(root) {
    const maxRate = (function walk(n) { return Math.max(n.rate, n.leaf ? 0 : Math.max(walk(n.left), walk(n.right))); })(root) || 1;
    const node = (n, branch) => {
      const hasKids = !n.leaf;
      const kids = hasKids ? `<ul>${node(n.left, `${n.label} ≤ ${n.threshold}`)}${node(n.right, `${n.label} > ${n.threshold}`)}</ul>` : "";
      return `<li><span class="node ${n.active ? "active" : ""}" ${hasKids ? 'data-toggle="1"' : ""} title="${n.days} days, ${(n.rate * 100).toFixed(1)}% followed by an unusual move">
        <span class="toggle">${hasKids ? "▾" : ""}</span>
        <span>${esc(branch)}</span>
        <span class="bar"><span style="width:${(n.rate / maxRate) * 100}%"></span></span>
        <span class="stat">${(n.rate * 100).toFixed(0)}% · ${n.days}d</span>
        ${n.leaf ? '<span class="leafmark">leaf</span>' : ""}
      </span>${kids}</li>`;
    };
    $("#tree").innerHTML = `<ul>${node(root, "All days")}</ul>`;
  }

  function anomalyKind(a) {
    if (/spike|crash|gap|range/i.test(a.type)) return "price";
    if (/volume/i.test(a.type)) return "volume";
    return "outlier";
  }

  function renderTable() {
    const d = state.detail; if (!d) return;
    const { key, dir } = state.tableSort;
    let rows = d.anomalies.filter((a) => state.tableFilter === "all" || anomalyKind(a) === state.tableFilter || (state.tableFilter === "volume" && /volume/i.test(a.type)));
    rows = [...rows].sort((a, b) => {
      const x = a[key], y = b[key];
      if (typeof x === "string") return dir * x.localeCompare(y);
      return dir * ((x ?? -Infinity) - (y ?? -Infinity));
    });
    const cols = [["date", "Date"], ["type", "Type"], ["ret_pct", "Move"], ["ret_z", "Move σ"], ["vol_z", "Vol σ"], ["score", "Score"]];
    const arrow = (k) => (k === key ? (dir > 0 ? " ↑" : " ↓") : "");
    const sym = (a) => (/spike/i.test(a.type) ? `<span class="sym" style="color:var(--up)">▲</span>` : /crash/i.test(a.type) ? `<span class="sym" style="color:var(--down)">▼</span>` : `<span class="sym" style="color:var(--other)">●</span>`);
    $("#anom-table").innerHTML = rows.length ? `<table><thead><tr>${cols.map(([k, l]) => `<th data-k="${k}">${l}${arrow(k)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((a) => `<tr data-date="${a.date}"><td>${a.date}</td><td>${sym(a)} ${esc(a.type)}</td><td class="${a.ret_pct >= 0 ? "up" : "down"}">${signed(a.ret_pct)}%</td><td>${signed(a.ret_z, 1)}</td><td>${a.vol_z == null ? "–" : signed(a.vol_z, 1)}</td><td>${Math.round(a.score)}</td></tr>`).join("")}</tbody></table>`
      : `<p class="muted small">No anomalies of this kind.</p>`;
  }

  function bindDetailEvents() {
    $("#price-range").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      state.priceRange = b.dataset.r;
      $$("#price-range button").forEach((x) => x.classList.toggle("active", x === b));
      drawPrice(state.detail);
    });
    $("#table-filter").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      state.tableFilter = b.dataset.f;
      $$("#table-filter button").forEach((x) => x.classList.toggle("active", x === b));
      renderTable();
    });
    $("#anom-table").addEventListener("click", (e) => {
      const th = e.target.closest("th");
      if (th) {
        const k = th.dataset.k;
        state.tableSort = { key: k, dir: state.tableSort.key === k ? -state.tableSort.dir : -1 };
        renderTable(); return;
      }
      const tr = e.target.closest("tr[data-date]");
      if (tr) {
        $$("#price-range button").forEach((x) => x.classList.remove("active"));
        drawPrice(state.detail, tr.dataset.date);
        $("#price-chart").scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
    $("#tree").addEventListener("click", (e) => {
      const n = e.target.closest(".node[data-toggle]"); if (!n) return;
      const li = n.parentElement;
      li.classList.toggle("collapsed");
      n.querySelector(".toggle").textContent = li.classList.contains("collapsed") ? "▸" : "▾";
    });
  }

  // ---------- controls ----------
  function syncOutputs() {
    $("#threshold-out").textContent = `${(+$("#threshold").value).toFixed(1)}σ`;
    const h = +$("#horizon").value;
    $("#horizon-out").textContent = `${h} day${h > 1 ? "s" : ""}`;
    $("#depth-out").textContent = $("#depth").value;
  }

  let debounce;
  function settingsChanged() {
    syncOutputs();
    store.set("anomality.settings", Object.fromEntries(params()));
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      if (state.selected) select(state.selected);
      scan();
    }, 450);
  }

  function addSymbol(raw) {
    const sym = raw.trim().toUpperCase();
    if (!sym) return;
    if (!allSymbols().includes(sym)) {
      state.custom.push(sym);
      store.set("anomality.custom", state.custom);
      scan([sym]);
    }
    select(sym);
  }

  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  }

  function init() {
    const saved = store.get("anomality.settings", null);
    if (saved) ["period", "threshold", "horizon", "model", "depth"].forEach((k) => { if (saved[k] != null) $(`#${k}`).value = saved[k]; });
    applyTheme(store.get("anomality.theme", null));
    syncOutputs();

    ["period", "model"].forEach((id) => $(`#${id}`).addEventListener("change", settingsChanged));
    ["threshold", "horizon", "depth"].forEach((id) => {
      $(`#${id}`).addEventListener("input", syncOutputs);
      $(`#${id}`).addEventListener("change", settingsChanged);
    });
    $("#rescan").addEventListener("click", () => { scan(); if (state.selected) select(state.selected); });
    $("#sort").addEventListener("change", renderWatchlist);
    $("#search").addEventListener("submit", (e) => { e.preventDefault(); addSymbol($("#search-input").value); $("#search-input").value = ""; });
    $$(".tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.group === state.group);
      t.addEventListener("click", () => {
        state.group = t.dataset.group; store.set("anomality.group", state.group);
        $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
        renderWatchlist();
      });
    });
    $("#watchlist").addEventListener("click", (e) => {
      const rm = e.target.closest("[data-remove]");
      if (rm) {
        e.stopPropagation();
        state.custom = state.custom.filter((s) => s !== rm.dataset.remove);
        store.set("anomality.custom", state.custom);
        renderWatchlist(); return;
      }
      const li = e.target.closest("li[data-sym]");
      if (li) select(li.dataset.sym);
    });
    $("#theme-toggle").addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme
        ? document.documentElement.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      const next = dark ? "light" : "dark";
      applyTheme(next); store.set("anomality.theme", next);
      if (state.detail && $("#price-chart")) renderDetail(state.detail);
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (state.detail && $("#price-chart")) renderDetail(state.detail); });

    api("/api/watchlist").then((w) => {
      state.lists = { stocks: w.stocks, crypto: w.crypto };
      renderWatchlist();
      scan();
      select(state.selected && allSymbols().includes(state.selected) ? state.selected : w.stocks[0]);
    }).catch((e) => { $("#detail").innerHTML = `<div class="error">Cannot reach the anomality server: ${esc(e.message)}</div>`; });
  }

  init();
})();
