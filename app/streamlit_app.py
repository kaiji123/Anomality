from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.dashboard_utils import REGIME_COLORS, SYSTEM_COLORS, compute_metrics, has_run_system, regime_intervals, truncate_at_date, verdict
from app.demo_data import generate_demo_data
from src.adaptivetrader.dashboard_io import (
    DashboardDataError,
    load_backtest,
    load_decisions,
    load_events,
    load_lab,
    load_market,
    load_regimes,
    load_runs,
    load_summary,
)


def _warn_demo():
    st.warning("DEMO DATA — not real results")


def _source_label(name: str, demo_used: set[str]) -> str:
    return "DEMO" if name in demo_used else "REAL"


def _load_data():
    rd = Path(os.getenv("RESULTS_DIR", str(ROOT / "results")))
    errors = {}
    data = {}
    loaders = {
        "market": load_market,
        "regimes": load_regimes,
        "backtest": load_backtest,
        "events": load_events,
        "decisions": load_decisions,
        "summary": load_summary,
        "lab": load_lab,
        "runs": load_runs,
    }
    for k, fn in loaders.items():
        try:
            data[k] = fn(str(rd))
        except DashboardDataError as exc:
            data[k] = None
            errors[k] = str(exc)

    demo = generate_demo_data()
    demo_used = set()
    for k in loaders:
        if data[k] is None:
            data[k] = demo[k]
            demo_used.add(k)
    return data, errors, demo_used


def _iso(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _sidebar(data, demo_used, errors):
    st.sidebar.header("Data status")
    st.sidebar.write(f"Data source: **{'DEMO' if len(demo_used) else 'REAL'}**")
    st.sidebar.caption("Per-file source")
    for k in ["market", "regimes", "backtest", "events", "decisions", "summary", "lab", "runs"]:
        st.sidebar.write(f"- {k}: {_source_label(k, demo_used)}")

    runs = data["runs"]
    if runs is not None and not runs.empty:
        latest = runs.sort_values("timestamp").iloc[-1]
        st.sidebar.write(f"run_id: `{latest['run_id']}`")
        st.sidebar.write(f"git_commit: `{latest['git_commit']}`")

    mk = data["market"]
    if mk is not None and not mk.empty:
        st.sidebar.write(f"Evaluation period: {_iso(mk['date'].min())} → {_iso(mk['date'].max())}")
    if errors:
        st.sidebar.error("Validation issues")
        for e in errors.values():
            st.sidebar.caption(e)


def _filters(market):
    assets = sorted(market["asset"].unique())
    default = "SPY" if "SPY" in assets else assets[0]
    asset = st.selectbox("Asset", assets, index=assets.index(default))
    dates = sorted(pd.to_datetime(market[market["asset"] == asset]["date"]).dt.date.unique())
    default_date = pd.Timestamp("2020-02-14").date()
    if "tm_idx" not in st.session_state:
        st.session_state.tm_idx = dates.index(default_date) if default_date in dates else len(dates) - 1
    sel = st.slider("Date", min_value=dates[0], max_value=dates[-1], value=dates[st.session_state.tm_idx])
    st.session_state.tm_idx = dates.index(sel)
    c1, c2, c3 = st.columns(3)
    with c1:
        playing = st.toggle("Play", key="tm_play")
    with c2:
        if st.button("Step +1 day"):
            st.session_state.tm_idx = min(st.session_state.tm_idx + 1, len(dates) - 1)
            st.rerun()
    with c3:
        if st.button("Step -1 day"):
            st.session_state.tm_idx = max(st.session_state.tm_idx - 1, 0)
            st.rerun()

    if hasattr(st, "fragment"):
        @st.fragment(run_every=1 if playing else None)
        def _tick():
            if st.session_state.get("tm_play"):
                st.session_state.tm_idx = min(st.session_state.tm_idx + 1, len(dates) - 1)
                if st.session_state.tm_idx == len(dates) - 1:
                    st.session_state.tm_play = False
                st.rerun()
        _tick()

    return asset, pd.Timestamp(dates[st.session_state.tm_idx])


def page_time_machine(data, demo_used, errors):
    st.title("Time machine")
    if any(k in demo_used for k in ["market", "regimes", "backtest", "events", "decisions"]):
        _warn_demo()
    for k in ["market", "regimes", "backtest", "events", "decisions"]:
        if k in errors:
            st.error(errors[k])

    market, regimes, backtest, events, decisions = data["market"], data["regimes"], data["backtest"], data["events"], data["decisions"]
    if market is None or regimes is None or backtest is None:
        st.info("No market/regime/backtest file yet — waiting for adaptive engine output")
        return

    asset, cutoff = _filters(market)
    st.caption(f"Showing data through {_iso(cutoff)}")
    m = truncate_at_date(market[market["asset"] == asset], "date", cutoff)
    r = truncate_at_date(regimes[regimes["asset"] == asset], "date", cutoff)
    b = truncate_at_date(backtest[backtest["asset"] == asset], "date", cutoff)
    e = truncate_at_date(events[events["asset"] == asset], "date", cutoff) if events is not None else pd.DataFrame()
    d = truncate_at_date(decisions[decisions["asset"] == asset], "date", cutoff) if decisions is not None else pd.DataFrame()

    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=m["date"], y=m["close"], name="Close", mode="lines", line={"color": "#222"}))
    for iv in regime_intervals(r):
        pfig.add_vrect(x0=iv["start"], x1=iv["end"], fillcolor=REGIME_COLORS.get(iv["regime"], "rgba(120,120,120,0.1)"), opacity=0.35, line_width=0)
    if not e.empty:
        em = e.merge(m[["date", "close"]], on="date", how="left")
        pfig.add_trace(go.Scatter(x=em["date"], y=em["close"], mode="markers", marker={"size": 8, "color": "#d62728"}, name="Events", text=em["event_type"]))
    pfig.update_layout(height=340, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(pfig, use_container_width=True)

    efig = go.Figure()
    for s in ["A", "B", "C", "BH"]:
        part = b[b["system"] == s]
        if not part.empty:
            efig.add_trace(go.Scatter(x=part["date"], y=part["equity"], mode="lines", name=s, line={"color": SYSTEM_COLORS[s]}))
    efig.update_layout(height=300, xaxis_title="Date", yaxis_title="Equity")
    st.plotly_chart(efig, use_container_width=True)

    with st.container(border=True):
        ld = d.iloc[-1] if not d.empty else None
        le = e.iloc[-1] if not e.empty else None
        st.subheader("Current decision")
        st.write(f"Action: **{ld['action'] if ld is not None else 'n/a'}**")
        st.write(f"Active strategy/model: **{ld['active_strategy_or_model'] if ld is not None else 'n/a'}**")
        st.write(f"Reason: {ld['reason'] if ld is not None else 'n/a'}")
        st.write(f"Latest event: {le['event_type'] if le is not None else 'n/a'}")

    if has_run_system():
        st.button("Truncation check")


def page_strategy_lab(data, demo_used, errors):
    st.title("Strategy lab")
    if any(k in demo_used for k in ["backtest", "summary", "decisions"]):
        _warn_demo()
    for k in ["backtest", "summary", "decisions"]:
        if k in errors:
            st.error(errors[k])

    bt, summary, decisions = data["backtest"], data["summary"], data["decisions"]
    if bt is None or summary is None:
        st.info("No backtest/summary file yet — waiting for adaptive engine output")
        return

    costs = sorted(pd.to_numeric(bt["cost_bps"], errors="coerce").dropna().unique())
    cost = st.selectbox("cost_bps", costs)
    asset = st.selectbox("Asset", sorted(bt["asset"].unique()))
    f = bt[(bt["cost_bps"] == cost) & (bt["asset"] == asset)]

    eq, dd = go.Figure(), go.Figure()
    for s, p in f.groupby("system"):
        eq.add_trace(go.Scatter(x=p["date"], y=p["equity"], mode="lines", name=s, line={"color": SYSTEM_COLORS.get(s, "#444")}))
        dd.add_trace(go.Scatter(x=p["date"], y=p["drawdown"], mode="lines", name=s, line={"color": SYSTEM_COLORS.get(s, "#444")}))
    eq.update_layout(height=300, yaxis_title=f"Equity (net of {cost} bps)")
    dd.update_layout(height=250, yaxis_title=f"Drawdown (net of {cost} bps)")
    st.plotly_chart(eq, use_container_width=True)
    st.plotly_chart(dd, use_container_width=True)

    met = compute_metrics(f, decisions[decisions["asset"] == asset] if decisions is not None else None)
    if not met.empty:
        m2 = met.copy()
        m2["net_annual_return"] = (m2["net_annual_return"] * 100).round(2).astype(str) + "%/yr"
        m2["volatility"] = (m2["volatility"] * 100).round(2).astype(str) + "%/yr"
        m2["max_drawdown"] = (m2["max_drawdown"] * 100).round(2).astype(str) + "%"
        m2["turnover"] = m2["turnover"].round(4).astype(str) + " ratio/day"
        m2["switches_per_year"] = m2["switches_per_year"].round(2).astype(str) + " switches/yr"
        st.dataframe(m2, use_container_width=True)

    with st.expander("Metric definitions"):
        st.write("Net annual return: annualized compounded net equity return.")
        st.write("Volatility: annualized stdev of daily net returns.")
        st.write("Sharpe: annualized return/risk on daily net returns.")
        st.write("Max drawdown: minimum drawdown.")
        st.write("Turnover: mean daily turnover.")
        st.write("Switches/year: annualized switch actions.")

    s = summary[summary["cost_bps"] == cost].copy()
    s["verdict"] = s.apply(lambda x: verdict(float(x["ci_low"]), float(x["ci_high"])), axis=1)
    cfig = go.Figure()
    cfig.add_trace(go.Scatter(
        x=s["asset"], y=s["delta_sharpe"], mode="markers", marker={"size": 10, "color": "#7b3fe4"},
        error_y={"type": "data", "array": s["ci_high"] - s["delta_sharpe"], "arrayminus": s["delta_sharpe"] - s["ci_low"]}
    ))
    cfig.update_layout(height=320, yaxis_title=f"ΔSharpe (net of {cost} bps)")
    st.plotly_chart(cfig, use_container_width=True)
    pooled = s[s["asset"] == "POOLED"]
    if not pooled.empty:
        row = pooled.iloc[0]
        st.markdown(f"**POOLED verdict:** `{verdict(float(row['ci_low']), float(row['ci_high']))}` (ci_low={row['ci_low']:.4f}, ci_high={row['ci_high']:.4f})")


def page_learning(data, demo_used, errors):
    st.title("Learning timeline")
    if any(k in demo_used for k in ["events", "market"]):
        _warn_demo()
    if "events" in errors:
        st.error(errors["events"])
    events, market = data["events"], data["market"]
    if events is None or events.empty:
        st.info("No events file yet — waiting for adaptive engine output")
        return

    a = st.selectbox("Asset filter", ["ALL"] + sorted(events["asset"].unique()))
    t = st.selectbox("Event type filter", ["ALL"] + sorted(events["event_type"].unique()))
    f = events.copy()
    if a != "ALL":
        f = f[f["asset"] == a]
    if t != "ALL":
        f = f[f["event_type"] == t]
    fig = go.Figure([go.Scatter(x=f["date"], y=f["event_type"], mode="markers")])
    fig.update_layout(height=240)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(f, use_container_width=True)

    if not f.empty:
        eid = st.selectbox("Select event", f["event_id"].astype(str).tolist())
        row = f[f["event_id"].astype(str) == eid].iloc[0]
        st.write(f"Metric before/after: {row.get('metric_before','n/a')} → {row.get('metric_after','n/a')}")
        if market is not None and not market.empty:
            mm = market[market["asset"] == row["asset"]].sort_values("date")
            if (mm["date"] == row["date"]).any():
                i = mm.index[mm["date"] == row["date"]][0]
                loc = mm.index.get_loc(i)
                win = mm.iloc[max(0, loc - 30): min(len(mm), loc + 31)]
                wf = go.Figure([go.Scatter(x=win["date"], y=win["close"], mode="lines")])
                wf.add_vline(x=row["date"], line_dash="dash", line_color="#d62728")
                st.plotly_chart(wf, use_container_width=True)


def page_risk(data, demo_used, errors):
    st.title("Risk & decisions")
    if any(k in demo_used for k in ["decisions", "lab"]):
        _warn_demo()
    if "decisions" in errors:
        st.error(errors["decisions"])
    if "lab" in errors:
        st.error(errors["lab"])

    d, lab = data["decisions"], data["lab"]
    if d is None or d.empty:
        st.info("No decisions file yet — waiting for adaptive engine output")
        return

    t1, t2 = st.tabs(["Decision timeline", "Stress lab"])
    with t1:
        asset = st.selectbox("Asset", sorted(d["asset"].unique()), key="risk_asset")
        x = d[d["asset"] == asset].sort_values("date")
        fig = go.Figure([go.Scatter(x=x["date"], y=x["position"], mode="lines+markers", name="position")])
        if "prob_up" in x.columns:
            fig.add_trace(go.Scatter(x=x["date"], y=x["prob_up"], mode="lines", name="prob_up", yaxis="y2"))
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 1]})
        st.plotly_chart(fig, use_container_width=True)
        cols = ["date", "system", "action", "position", "active_strategy_or_model", "reason"] + (["prob_up"] if "prob_up" in x.columns else [])
        st.dataframe(x[cols], use_container_width=True)
    with t2:
        st.warning("SYNTHETIC")
        if lab is None or lab.empty:
            st.info("No lab.parquet file yet")
        else:
            lf = go.Figure()
            for s, p in lab.groupby("system"):
                lf.add_trace(go.Scatter(x=p["detection_delay_days"], y=p["false_switches_per_year"], mode="markers", name=s, marker={"color": SYSTEM_COLORS.get(s, "#444")}))
            st.plotly_chart(lf, use_container_width=True)
            st.dataframe(lab[lab["path_id"] == lab["path_id"].iloc[0]], use_container_width=True)


def main():
    st.set_page_config(page_title="AdaptiveTrader dashboard", layout="wide")
    data, errors, demo_used = _load_data()
    _sidebar(data, demo_used, errors)
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        nav = st.navigation([
            st.Page(lambda: page_time_machine(data, demo_used, errors), title="Time machine"),
            st.Page(lambda: page_strategy_lab(data, demo_used, errors), title="Strategy lab"),
            st.Page(lambda: page_learning(data, demo_used, errors), title="Learning timeline"),
            st.Page(lambda: page_risk(data, demo_used, errors), title="Risk & decisions"),
        ])
        nav.run()
    else:
        pg = st.sidebar.radio("Page", ["Time machine", "Strategy lab", "Learning timeline", "Risk & decisions"])
        if pg == "Time machine":
            page_time_machine(data, demo_used, errors)
        elif pg == "Strategy lab":
            page_strategy_lab(data, demo_used, errors)
        elif pg == "Learning timeline":
            page_learning(data, demo_used, errors)
        else:
            page_risk(data, demo_used, errors)


if __name__ == "__main__":
    main()
