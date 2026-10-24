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
    market: null,        // /api/regime/market (core feature)
    regime: {},          // symbol -> /api/regime/{symbol}
    regimeRange: "All",
    heroChart: false,
    showOriginal: false,
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
  const signed = (v, d = 2) => {
    if (v == null) return "–";
    const s = v.toFixed(d);
    if (Number(s) === 0) return (0).toFixed(d); // avoid "-0.0"
    return `${v > 0 ? "+" : ""}${s}`;
  };
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

  // ---------- regime model helpers ----------
  const WEEK = Math.sqrt(5 / 252);
  function moodBadge(s) {
    const m = state.market?.technical?.assets?.[s]?.mood;
    const icon = { Calm: "●", Normal: "◆", Choppy: "▲", Stormy: "⚠" }[m];
    return m ? `<span class="mood-sm ${m}" title="This week's mood for ${esc(s)}, compared with its own history">${icon} ${m}</span>` : "";
  }
  const volOf = (s) => state.market?.technical?.assets?.[s]?.forecast;
  function volPill(s) {
    const a = state.market?.technical?.assets?.[s];
    if (!a) return "";
    return `<span class="vol-pill" title="Expected move over the next week (crash-tested forecast)">±${(a.forecast * WEEK * 100).toFixed(1)}% this week</span>`;
  }
  const REG_ORDER = ["Calm", "Normal", "Elevated", "Stress"];
  function weightBar(weights) {
    const ws = [...weights].sort((a, b) => REG_ORDER.indexOf(a.name) - REG_ORDER.indexOf(b.name));
    const bar = ws.map((w) => `<span class="r${REG_ORDER.indexOf(w.name)}" style="flex:${w.weight}" title="${esc(w.name)} expert: ${(w.weight * 100).toFixed(0)}%">${w.weight >= 0.14 ? `${esc(w.name)} ${(w.weight * 100).toFixed(0)}%` : ""}</span>`).join("");
    return `<div class="wbar" role="img" aria-label="Expert weights: ${ws.map((w) => `${w.name} ${(w.weight * 100).toFixed(0)}%`).join(", ")}">${bar}</div>`;
  }
  const pctv = (v, d = 0) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);

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
      volfc: (a, b) => (volOf(b) ?? -1) - (volOf(a) ?? -1),
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
          <div class="wl-name" title="${esc(r.name)}">${esc(r.name)}</div>${volPill(s)}</div>
        ${sparkline(r.spark, up)}
        <div class="wl-right"><div>${money(r.price, r.currency)}</div>
          <div class="${up ? "up" : "down"}">${signed(r.change_pct)}%</div>
          <div style="margin-top:3px">${moodBadge(s) || `<span title="Classic risk score ${r.risk_score}">${badge(r.verdict)}</span>`}</div></div>
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
    loadRegime(sym);
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
      <div id="regime-card"></div>
      ${s.synthetic ? `<div class="banner">⚠ Live data could not be fetched, so this shows <b>synthetic demo data</b>. Check your internet connection.</div>` : ""}

      <div class="grid top">
        <div class="card verdict">
          <h2>Classic model verdict</h2>
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
    renderRegimeCard(s.symbol.toUpperCase());
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

  // ---------- core feature: crisis-robust regime model (plain-language view) ----------
  const DOCS_URL = "https://github.com/kaiji123/Anomality/blob/master/docs/CRISIS_ROBUST_GATING.md";
  const MOOD_ICON = { Calm: "●", Normal: "◆", Choppy: "▲", Stormy: "⚠" };
  const moodChip = (m) => (m ? `<span class="mood ${m.name}">${MOOD_ICON[m.name] || ""} ${esc(m.name)}</span>` : "");

  // Mood scale: four equal segments with their weekly-move ranges and a marker for this week.
  function moodScale(m, scope = "market") {
    const sc = m?.scale;
    if (!sc || sc.value == null) return "";
    const edges = [sc.low, ...sc.cuts, sc.high];
    const i = sc.cuts.filter((c) => sc.value >= c).length;
    const frac = Math.max(0.04, Math.min(0.96, (sc.value - edges[i]) / ((edges[i + 1] - edges[i]) || 1)));
    const pos = ((i + frac) / sc.names.length) * 100;
    const f = (v) => v.toFixed(1);
    const ranges = [`under ±${f(sc.cuts[0])}%`, `±${f(sc.cuts[0])}–${f(sc.cuts[1])}%`, `±${f(sc.cuts[1])}–${f(sc.cuts[2])}%`, `over ±${f(sc.cuts[2])}%`];
    const p = Math.round(m.percentile * 100);
    const since = (sc.since || "").slice(0, 4);
    const compare = p >= 50 ? `busier than <b>${p}%</b> of weeks since ${since}` : `calmer than <b>${100 - p}%</b> of weeks since ${since}`;
    return `<div class="mscale" role="img" aria-label="Mood scale: ${sc.names.map((n, k) => `${n} ${ranges[k]}`).join(", ")}. This week ±${f(sc.value)}%, ${m.name}.">
      <div class="ms-track">
        ${sc.names.map((n, k) => `<span class="ms ${n} ${k === i ? "on" : ""}" title="${esc(n)} (${ranges[k]}): ${esc(m.levels?.[k]?.text || "")}">${n}</span>`).join("")}
        <span class="ms-marker" style="left:${pos}%"><b>±${f(sc.value)}%</b></span>
      </div>
      <div class="ms-ranges">${ranges.map((r) => `<span>${r}</span>`).join("")}</div>
      <div class="ms-note">Expected move this week is ${compare}.</div>
      <button class="link-btn small mood-help" type="button" data-scope="${esc(scope)}">What do the levels mean?</button>
    </div>`;
  }

  function moodRanges(sc) {
    const f = (v) => v.toFixed(1);
    return [`under ±${f(sc.cuts[0])}%`, `±${f(sc.cuts[0])}–${f(sc.cuts[1])}%`, `±${f(sc.cuts[1])}–${f(sc.cuts[2])}%`, `over ±${f(sc.cuts[2])}%`];
  }

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  // ["2020-03","2020-04","2022-05"] -> ["Mar–Apr 2020", "May 2022"] (most recent first)
  function monthRanges(list, max = 3) {
    const idx = list.map((m) => { const [y, mo] = m.split("-").map(Number); return y * 12 + mo - 1; }).sort((a, b) => a - b);
    const groups = [];
    for (const v of idx) {
      const g = groups[groups.length - 1];
      if (g && v === g[1] + 1) g[1] = v; else groups.push([v, v]);
    }
    const fmt = (v) => [MONTHS[v % 12], Math.floor(v / 12)];
    return groups.reverse().slice(0, max).map(([a, b]) => {
      const [ma, ya] = fmt(a), [mb, yb] = fmt(b);
      if (a === b) return `${ma} ${ya}`;
      return ya === yb ? `${ma}–${mb} ${ya}` : `${ma} ${ya}–${mb} ${yb}`;
    });
  }

  function moodLegend(m, ranges, opts = {}) {
    if (!m.levels) return "";
    const rows = m.levels.map((l, k) => {
      const when = l.months.length ? monthRanges(l.months).join(", ") : "not yet seen";
      return `<li class="${l.name === m.name ? "now" : ""}">
        <span class="mood-sm ${l.name}">${MOOD_ICON[l.name]} ${esc(l.name)}</span>
        <div class="lg-body">
          <div class="lg-top"><b>${ranges[k]}</b> expected move · ${Math.round(l.share * 100)}% of weeks${l.name === m.name ? ' · <span class="lg-now">this week</span>' : ""}</div>
          <div class="lg-text">${esc(l.text)}</div>
          ${l.actual_avg != null ? `<div class="lg-stat">In these weeks prices actually moved <b>±${l.actual_avg.toFixed(1)}%</b> on average, up to <b>±${l.actual_rough.toFixed(1)}%</b> in rough weeks.</div>` : ""}
          <div class="lg-stat">Most recently: ${esc(when)}</div>
        </div>
      </li>`;
    }).join("");
    return `<ul class="lg-list">${rows}</ul>
      <p class="lg-foot">Levels compare the model's forecast with its own forecasts since ${(m.scale?.since || "").slice(0, 4)}: Calm is the quietest 35% of weeks, Normal the next 35%, Choppy the next 20%, Stormy the busiest 10%. "Rough weeks" = the worst 1 in 10.</p>`;
  }
  const pm = (v, d = 1) => (v == null ? "–" : `±${Number(v).toFixed(d)}%`);
  const lessPct = (a, b) => Math.round((1 - a / b) * 100);

  function sigRow(label, cls, v, max) {
    const w = Math.min(100, (Math.abs(v) / max) * 100);
    return `<div class="sig-row"><span>${label}</span><div class="sig-track"><div class="sig-band" style="width:${(2 / max) * 100}%"></div><div class="sig-fill ${cls}" style="width:${w}%"></div></div><span class="sig-val">${Math.abs(v).toFixed(1)}×</span></div>`;
  }

  function renderHero(m) {
    const el = $("#regime-hero");
    const t = m.track, tech = m.technical, rs = t?.research;
    const covid = tech.crisis_agreement?.["COVID crash"], calm = tech.crisis_agreement?.["Calm 2019"];
    const assetLink = (a) => `<li><button class="link-btn" data-sym="${esc(a.symbol)}">${esc(a.symbol)}</button><b>${pm(a.move)}</b></li>`;
    const sgn = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}σ`;
    el.innerHTML = `
      <div class="hero-head">
        <span class="core-badge">★ Core feature</span>
        <h2>This week's anomaly report</h2>
        <span class="muted">How much prices could move this week, from a crash-tested AI model · data to ${esc(m.date)}</span>
      </div>
      <div class="hero-grid">
        <div class="hero-tile">
          <div class="label">Market mood</div>
          <div class="big">${moodChip(m.mood)}</div>
          <div class="note">${esc(m.mood.text)}</div>
          ${moodScale(m.mood, "market")}
        </div>
        <div class="hero-tile">
          <div class="label">Expected move this week</div>
          <div class="moves">
            <div><span class="mv">${pm(m.stock_move)}</span><span class="mvl">typical stock</span></div>
            ${m.crypto_move != null ? `<div><span class="mv">${pm(m.crypto_move)}</span><span class="mvl">typical crypto</span></div>` : ""}
          </div>
          <div class="note">${m.usual_stock_move ? `A usual week for stocks is ${pm(m.usual_stock_move)}. ` : ""}About 2 weeks in 3 stay inside this range.</div>
        </div>
        <div class="hero-tile">
          <div class="label">Biggest expected swings</div>
          <ul class="movers">${m.biggest.slice(0, 3).map(assetLink).join("")}</ul>
          <div class="label" style="margin-top:8px">Calmest stocks</div>
          <ul class="movers">${m.calmest.slice(0, 2).map(assetLink).join("")}</ul>
        </div>
        <div class="hero-tile">
          <div class="label">Why you can trust it</div>
          ${t ? `<ul class="trust">
            <li><b>${t.beats_simple} of ${t.periods}</b> test periods since ${t.since.slice(0, 4)}: closer than the simple "last month" estimate</li>
            <li>Assets it flags as most volatile moved <b>${(t.ranking.top / t.ranking.bottom).toFixed(1)}× more</b> than the ones it flags as calm</li>
            <li>In the 2020 crash, its misses were <b>${lessPct(t.covid_miss.ours, t.covid_miss.simple)}% smaller</b> than the simple estimate's</li>
          </ul>` : `<div class="note">Run the walk-forward training to see the track record.</div>`}
        </div>
      </div>
      <details class="hero-more" id="report-more" ${state.heroChart ? "open" : ""}>
        <summary><span class="when-closed">Read more about the anomaly report</span><span class="when-open">Hide the full report</span></summary>

        <section class="report-sec">
          <h3>1. Why this report holds up in a crash</h3>
          <div class="hero-more-grid">
            <div>
              <p class="explain-big">Most forecasting models learn from normal markets, then get confused by a crash, exactly when you need them. We made ours <b>crash-proof</b> by measuring volatility on a log scale, so a sudden spike no longer looks impossibly "off the charts" to the model.</p>
            </div>
            <div>
              <div class="label" style="margin-bottom:2px">If a COVID-sized crash (${pctv(tech.covid_peak_z.market_vol)} market volatility) hit today, how "off the charts" would it look?</div>
              <div class="sig">
                ${sigRow('<i class="swatch raw"></i>Original', "raw", tech.covid_peak_z.raw, 12)}
                ${sigRow('<i class="swatch log"></i>Ours', "log", tech.covid_peak_z.log, 12)}
                <div class="sig-axis"><span>normal</span><span>familiar range</span><span>off the charts</span></div>
              </div>
              <p class="note" style="margin-top:6px">"Normal-sized steps" away from what the model learned. Original = the published research model (Ye &amp; Borde, 2026); ours adds the log scale, keeping a crash <b>${(tech.covid_peak_z.raw / tech.covid_peak_z.log).toFixed(1)}× closer</b> to familiar ground.</p>
            </div>
          </div>
        </section>

        ${t ? `<section class="report-sec">
          <h3>2. Track record</h3>
          <div class="hero-more-grid">
            <div>
              <div class="label" style="margin-bottom:4px">Average forecast miss on the weekly move (lower is better)</div>
              <table class="compare plain"><thead><tr><th></th><th>All weeks since ${t.since.slice(0, 4)}</th><th>2020 crash</th></tr></thead><tbody>
                <tr><td><i class="swatch log"></i>Ours</td><td class="best">${pm(t.miss.ours, 2)}</td><td class="best">${pm(t.covid_miss.ours, 2)}</td></tr>
                <tr><td><i class="swatch raw"></i>Original research model</td><td>${pm(t.miss.original, 2)}</td><td>${pm(t.covid_miss.original, 2)}</td></tr>
                <tr><td>Simple estimate (last month)</td><td>${pm(t.miss.simple, 2)}</td><td>${pm(t.covid_miss.simple, 2)}</td></tr>
              </tbody></table>
              <ul class="trust" style="margin-top:10px">
                <li>Closer than the simple estimate in <b>${t.beats_simple} of ${t.periods}</b> test periods, and than the original model in <b>${t.beats_original} of ${t.periods}</b></li>
                <li>Its top-20% "most volatile" picks moved <b>${pm(t.ranking.top)}</b> a week on average; its bottom 20% moved <b>${pm(t.ranking.bottom)}</b></li>
                <li>When it says "calmer ahead", it's right <b>${Math.round(t.calmer_hit_rate * 100)}%</b> of the time</li>
              </ul>
            </div>
            <div>
              <div class="feature-legend"><span><i style="background:var(--accent)"></i>Forecast</span><span><i style="background:var(--ink-2)"></i>What actually happened</span><span class="muted">Typical stock, weekly move, test data the model never saw</span></div>
              <div id="hero-chart" style="height:250px"></div>
            </div>
          </div>
        </section>` : ""}

        <section class="report-sec">
          <h3>3. Advanced: inside the model</h3>
          <div class="adv-tiles">
            <div class="hero-tile">
              <div class="label">How the gate sees today's market</div>
              <div class="kv">
                <span class="k"><i class="swatch raw"></i>Original (raw)</span><span class="v">${sgn(tech.gate_input_z.raw)}</span>
                <span class="k"><i class="swatch log"></i>Ours (log)</span><span class="v">${sgn(tech.gate_input_z.log)}</span>
              </div>
              <div class="note" style="margin-top:6px">Market volatility ${pctv(tech.market_vol, 1)}, in standard deviations from the training average. Within ±2σ is familiar territory.</div>
            </div>
            <div class="hero-tile">
              <div class="label">Regime gate weights today</div>
              ${weightBar(tech.weights)}
              <div class="note" style="margin-top:6px">How much the model leans on each of its 4 experts, averaged across stocks. Near-equal weights mean no strong regime signal.</div>
            </div>
            <div class="hero-tile">
              <div class="label">Herding in the 2020 crash</div>
              ${covid ? `<div class="kv">
                <span class="k"><i class="swatch raw"></i>Original</span><span class="v">${pctv(covid.agree_raw)}</span>
                <span class="k"><i class="swatch log"></i>Ours</span><span class="v">${pctv(covid.agree_log)}</span>
                <span class="k">Calm 2019 (both)</span><span class="v">${pctv(calm?.agree_log)}</span></div>
              <div class="note" style="margin-top:6px">Share of stocks pushed into the same expert. Lower = the model keeps judging stocks individually under stress.</div>` : `<div class="note">Needs walk-forward history.</div>`}
            </div>
            <div class="hero-tile">
              <div class="label">Research result: log vs. raw inputs</div>
              ${rs ? `<div class="kv">
                <span class="k">Test windows improved</span><span class="v">${rs.windows_better} / ${rs.windows}</span>
                <span class="k">Ranking accuracy (IC)</span><span class="v">${rs.ic_raw.toFixed(3)} → ${rs.ic_log.toFixed(3)}</span>
                <span class="k">Significance</span><span class="v">p ${rs.p_value < 0.001 ? "< 0.001" : "= " + rs.p_value.toFixed(3)}</span></div>
              <div class="note" style="margin-top:6px">IC of the 3-seed average forecast; p from weekly-averaged daily IC differences. <a class="hero-link" href="${DOCS_URL}" target="_blank" rel="noopener">Full research write-up ↗</a></div>` : ""}
            </div>
          </div>
          ${tech.history ? `<div class="feature-legend" style="margin-top:12px"><span><i style="background:var(--raw)"></i>Original (raw inputs)</span><span><i style="background:var(--accent)"></i>Ours (log inputs)</span><span class="muted">Herding: share of stocks routed to the same expert, 21-day average</span></div>
          <div id="herding-chart" style="height:200px"></div>` : ""}
        </section>
      </details>`;
    el.hidden = false;
    $("#report-more").addEventListener("toggle", (e) => {
      state.heroChart = e.target.open;
      store.set("anomality.heroChart", state.heroChart);
      if (state.heroChart) drawReportCharts(m);
    });
    el.querySelectorAll(".link-btn[data-sym]").forEach((b) => b.addEventListener("click", () => {
      const s = b.dataset.sym;
      if (!allSymbols().includes(s)) { state.custom.push(s); store.set("anomality.custom", state.custom); scan([s]); }
      select(s);
      $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    if (state.heroChart) drawReportCharts(m);
  }

  function drawReportCharts(m) {
    if (m.history && $("#hero-chart")) drawHeroChart(m.history);
    const h = m.technical.history;
    if (!h || !$("#herding-chart")) return;
    const pct = (a) => a.map((v) => (v == null ? null : v * 100));
    Plotly.react("herding-chart", [
      { type: "scatter", mode: "lines", x: h.date, y: pct(smooth(h.agree_raw, 21)), line: { color: css("--raw"), width: 2 }, hovertemplate: "Original: %{y:.0f}% in one expert<extra></extra>" },
      { type: "scatter", mode: "lines", x: h.date, y: pct(smooth(h.agree_log, 21)), line: { color: css("--accent"), width: 2 }, hovertemplate: "Ours: %{y:.0f}%<extra></extra>" },
    ], {
      ...plotBase(), margin: { l: 44, r: 8, t: 18, b: 24 },
      xaxis: axis({ type: "date" }), yaxis: axis({ ticksuffix: "%" }),
      shapes: [{ type: "rect", xref: "x", yref: "paper", x0: "2020-02-19", x1: "2020-04-30", y0: 0, y1: 1, fillcolor: css("--grid"), opacity: 0.5, line: { width: 0 }, layer: "below" }],
      annotations: [{ xref: "x", yref: "paper", x: "2020-03-25", y: 1, text: "2020 crash", showarrow: false, yanchor: "bottom", font: { size: 10, color: css("--ink-2") } }],
    }, { ...plotCfg, displayModeBar: false });
  }

  function smooth(arr, n = 5) {
    return arr.map((_, i) => {
      const w = arr.slice(Math.max(0, i - n + 1), i + 1).filter((v) => v != null);
      return w.length ? w.reduce((a, b) => a + b, 0) / w.length : null;
    });
  }

  function drawHeroChart(h) {
    Plotly.react("hero-chart", [
      { type: "scatter", mode: "lines", x: h.date, y: h.actual, line: { color: css("--ink-2"), width: 1 }, opacity: 0.6, hovertemplate: "Actual ±%{y:.1f}%<extra></extra>" },
      { type: "scatter", mode: "lines", x: h.date, y: h.forecast, line: { color: css("--accent"), width: 2 }, hovertemplate: "Forecast ±%{y:.1f}%<extra></extra>" },
    ], {
      ...plotBase(), margin: { l: 44, r: 8, t: 18, b: 24 },
      xaxis: axis({ type: "date" }),
      yaxis: axis({ ticksuffix: "%", tickprefix: "±", rangemode: "tozero" }),
      shapes: [{ type: "rect", xref: "x", yref: "paper", x0: "2020-02-19", x1: "2020-04-30", y0: 0, y1: 1, fillcolor: css("--grid"), opacity: 0.5, line: { width: 0 }, layer: "below" }],
      annotations: [{ xref: "x", yref: "paper", x: "2020-03-25", y: 1, text: "2020 crash", showarrow: false, yanchor: "bottom", font: { size: 10, color: css("--ink-2") } }],
    }, { ...plotCfg, displayModeBar: false });
  }

  async function loadMarket() {
    try {
      state.market = await api("/api/regime/market");
      renderHero(state.market);
      renderWatchlist();
    } catch (e) {
      console.warn("Regime model unavailable:", e.message);
    }
  }

  const REGIME_RANGES = { "1Y": null, "3Y": null, All: null, "2020 crash": ["2020-01-15", "2020-07-31"], "2022 sell-off": ["2021-12-01", "2022-12-31"] };

  function renderRegimeCard(sym) {
    const el = $("#regime-card");
    if (!el) return;
    const r = state.regime[sym];
    const head = (extra = "") => `<div class="card-head"><div><span class="core-badge">★ Core feature</span> <h2 class="feature-title">Expected move this week</h2></div>${extra}</div>`;
    if (r === undefined) { el.innerHTML = `<div class="card feature">${head()}<p class="unavailable">Loading the forecast…</p></div>`; return; }
    if (r.error) { el.innerHTML = `<div class="card feature">${head()}<p class="unavailable">${esc(r.error)}</p></div>`; return; }
    const price = state.scan[sym]?.price ?? state.detail?.summary?.price;
    const cur = state.scan[sym]?.currency || "USD";
    const dollars = price ? price * r.move / 100 : null;
    const trendText = { calmer: "calmer than", wilder: "a bit livelier than", steady: "about the same as" }[r.trend];
    const rec = r.record, h = r.history, tech = r.technical;
    el.innerHTML = `
      <div class="card feature">
        ${head(`<a class="hero-link" href="${DOCS_URL}" target="_blank" rel="noopener">How it works ↗</a>`)}
        <div class="feature-hero">
          <div class="fh-main">
            <div class="fh-big">${pm(r.move)}</div>
            <div class="fh-sub">${dollars ? `about <b>±${money(dollars, cur)}</b> on today's ${money(price, cur)} price` : "typical one-week move"}</div>
          </div>
          <div class="fh-facts">
            <div><span class="k">Mood for ${esc(sym)}</span><span class="v">${moodChip(r.mood)}</span></div>
            ${r.usual_move ? `<div><span class="k">A usual week for ${esc(sym)}</span><span class="v">${pm(r.usual_move)}</span></div>` : ""}
            <div><span class="k">Last month it moved</span><span class="v">${pm(r.last_month_move)}</span></div>
          </div>
          <p class="fh-text">The model expects ${esc(sym)} to move about <b>${pm(r.move)}</b> over the next week, ${trendText} last month. About 2 weeks in 3 stay inside this range.</p>
          ${r.mood?.scale ? `<div class="fh-scale"><div class="label-sm">Mood scale for ${esc(sym)} (its own history)</div>${moodScale(r.mood, sym)}</div>` : ""}
        </div>
        ${h ? `
        <div class="card-head" style="align-items:center;margin-top:8px">
          <div class="feature-legend">
            <span><i style="background:var(--ink-2)"></i>What actually happened</span>
            <span><i style="background:var(--accent)"></i>Our forecast</span>
            <label class="cmp"><input type="checkbox" id="cmp-original" ${state.showOriginal ? "checked" : ""}> <i style="background:var(--raw)"></i>Compare with original research model</label>
            <span><i class="tri"></i>Surprise: moved 2×+ more than expected</span>
          </div>
          <div class="seg" id="regime-range">${Object.keys(REGIME_RANGES).map((k) => `<button data-r="${k}" class="${k === state.regimeRange ? "active" : ""}">${k}</button>`).join("")}</div>
        </div>
        <div id="regime-chart" style="height:290px"></div>
        <div class="feature-bottom">
          <div>
            <h3 class="mini-h">How accurate it has been for ${esc(sym)}</h3>
            <div class="record">
              <div><span class="rv">${pm(rec.ours, 1)}</span><span class="rl">average miss, ours</span></div>
              <div><span class="rv muted-v">${pm(rec.simple, 1)}</span><span class="rl">simple estimate</span></div>
              ${rec.covid ? `<div><span class="rv">${pm(rec.covid.ours, 1)}</span><span class="rl">in the 2020 crash<br>(simple: ${pm(rec.covid.simple, 1)}, original: ${pm(rec.covid.original, 1)})</span></div>` : ""}
            </div>
            <p class="small muted" style="margin:8px 0 0">"Miss" = how far the forecast weekly move was from what happened, on weeks the model never saw. One asset is noisy; across all 56 assets it beat the simple estimate in ${state.market?.track ? `${state.market.track.beats_simple} of ${state.market.track.periods}` : "most"} test periods.</p>
          </div>
          <div>
            <h3 class="mini-h">Surprises since ${h.date[0].slice(0, 4)} (${h.events.length})</h3>
            <p class="small muted" style="margin:0 0 4px">Times when ${esc(sym)} moved at least twice as much as forecast. Click one to see it on the chart.</p>
            <ul class="surprise-list" id="surprise-list">${h.events.slice(0, 40).map((e) => `<li data-date="${e.peak}"><span>${e.peak}</span><span>moved ${pm(e.actual)} vs. ${pm(e.forecast)} expected</span><b>${e.ratio.toFixed(1)}×</b></li>`).join("")}</ul>
          </div>
        </div>` : `<p class="unavailable">Run the walk-forward training to see the track record.</p>`}
        <details class="advanced">
          <summary>Advanced details</summary>
          <div class="adv-grid">
            <div><div class="label">Forecast, annualized volatility</div><div class="kv">
              <span class="k"><i class="swatch log"></i>Ours (log inputs)</span><span class="v">${pctv(tech.latest.forecast_log, 1)}</span>
              <span class="k"><i class="swatch raw"></i>Original (raw inputs)</span><span class="v">${pctv(tech.latest.forecast_raw, 1)}</span>
              <span class="k">Realized, last 20 days</span><span class="v">${pctv(tech.latest.realized_20d, 1)}</span></div></div>
            <div><div class="label">Regime gate weights today</div>${weightBar(tech.latest.weights)}
              <div class="note" style="margin-top:4px">Close to equal means no strong regime signal.</div></div>
            <div><div class="label">Model</div><div class="note">RG-ResMoE (Ye &amp; Borde, 2026) with log-scaled volatility inputs, ${tech.model.weights.toLocaleString()} weights, trained on 56 assets ${esc(tech.model.trained_from)} to ${esc(tech.model.trained_through)}.</div></div>
          </div>
        </details>
      </div>`;
    if (h) {
      drawRegimeChart(r);
      $("#regime-range").addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        state.regimeRange = b.dataset.r;
        $$("#regime-range button").forEach((x) => x.classList.toggle("active", x === b));
        drawRegimeChart(r);
      });
      $("#cmp-original").addEventListener("change", (e) => { state.showOriginal = e.target.checked; drawRegimeChart(r); });
      $("#surprise-list").addEventListener("click", (e) => {
        const li = e.target.closest("li[data-date]"); if (!li) return;
        $$("#regime-range button").forEach((x) => x.classList.remove("active"));
        drawRegimeChart(r, li.dataset.date);
      });
    }
  }

  function drawRegimeChart(r, focus) {
    const h = r.history;
    const last = h.date[h.date.length - 1];
    let range = REGIME_RANGES[state.regimeRange];
    if (!range) {
      const yrs = { "1Y": 1, "3Y": 3, All: 100 }[state.regimeRange] || 100;
      const d0 = new Date(last); d0.setFullYear(d0.getFullYear() - yrs);
      range = [d0 < new Date(h.date[0]) ? h.date[0] : d0.toISOString().slice(0, 10), last];
    }
    if (focus) {
      const d = new Date(focus), a = new Date(d), b = new Date(d);
      a.setDate(a.getDate() - 75); b.setDate(b.getDate() + 75);
      range = [a.toISOString().slice(0, 10), b.toISOString().slice(0, 10)];
    }
    const traces = [
      { type: "scatter", mode: "lines", x: h.date, y: h.actual, line: { color: css("--ink-2"), width: 1 }, opacity: 0.55, hovertemplate: "Actual ±%{y:.1f}%<extra></extra>" },
    ];
    if (state.showOriginal) traces.push({ type: "scatter", mode: "lines", x: h.date, y: h.original, line: { color: css("--raw"), width: 1.8 }, hovertemplate: "Original ±%{y:.1f}%<extra></extra>" });
    traces.push(
      { type: "scatter", mode: "lines", x: h.date, y: h.forecast, line: { color: css("--accent"), width: 2.2 }, hovertemplate: "Our forecast ±%{y:.1f}%<extra></extra>" },
      { type: "scatter", mode: "markers", x: h.events.map((e) => e.peak), y: h.events.map((e) => e.actual), marker: { symbol: "triangle-up", size: 9, color: css("--down"), line: { color: css("--surface"), width: 1 } }, hovertemplate: "Surprise: moved ±%{y:.1f}%<extra></extra>" },
    );
    Plotly.react("regime-chart", traces, {
      ...plotBase(), margin: { l: 48, r: 12, t: 8, b: 28 },
      xaxis: axis({ type: "date", range }),
      yaxis: axis({ ticksuffix: "%", tickprefix: "±", title: { text: "Weekly move", font: { size: 11 } }, rangemode: "tozero" }),
      shapes: focus ? [{ type: "line", xref: "x", yref: "paper", x0: focus, x1: focus, y0: 0, y1: 1, line: { color: css("--accent"), width: 1, dash: "dot" } }] : [],
    }, plotCfg);
  }

  async function loadRegime(sym) {
    if (state.regime[sym] && !state.regime[sym].error) { renderRegimeCard(sym); return; }
    renderRegimeCard(sym);
    try {
      state.regime[sym] = await api(`/api/regime/${encodeURIComponent(sym)}`);
    } catch (e) {
      state.regime[sym] = { error: e.message };
    }
    if (state.selected === sym) renderRegimeCard(sym);
  }

  // Levels explanation in a modal dialog
  function openLevels(scope) {
    const isMarket = scope === "market";
    const mood = isMarket ? state.market?.mood : state.regime[scope]?.mood;
    if (!mood?.levels || !mood.scale) return;
    $("#levels-modal-title").textContent = isMarket ? "What do the market mood levels mean?" : `What do the mood levels mean for ${scope}?`;
    const intro = isMarket
      ? `The market mood describes how much a typical stock is expected to move this week. This week: <b>±${mood.scale.value.toFixed(1)}%</b> (${esc(mood.name)}).`
      : `${esc(scope)}'s mood compares its expected move with its own history, because "normal" differs a lot between assets. This week: <b>±${mood.scale.value.toFixed(1)}%</b> (${esc(mood.name)}).`;
    $("#levels-modal-body").innerHTML = `<p class="modal-intro">${intro}</p>${moodLegend(mood, moodRanges(mood.scale))}`;
    const dlg = $("#levels-modal");
    if (typeof dlg.showModal === "function") dlg.showModal(); else dlg.setAttribute("open", "");
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
    document.addEventListener("click", (e) => {
      const b = e.target.closest(".mood-help");
      if (b) openLevels(b.dataset.scope);
    });
    const dlg = $("#levels-modal");
    $("#levels-modal-close").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); }); // click outside closes
    $("#theme-toggle").addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme
        ? document.documentElement.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      const next = dark ? "light" : "dark";
      applyTheme(next); store.set("anomality.theme", next);
      if (state.detail && $("#price-chart")) renderDetail(state.detail);
      if (state.market) renderHero(state.market);
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (state.detail && $("#price-chart")) renderDetail(state.detail); });

    state.heroChart = store.get("anomality.heroChart", false);
    loadMarket();
    api("/api/watchlist").then((w) => {
      state.lists = { stocks: w.stocks, crypto: w.crypto };
      renderWatchlist();
      scan();
      select(state.selected && allSymbols().includes(state.selected) ? state.selected : w.stocks[0]);
    }).catch((e) => { $("#detail").innerHTML = `<div class="error">Cannot reach the anomality server: ${esc(e.message)}</div>`; });
  }

  init();
})();
