"""Figures and tables from walk-forward results.

Usage: .venv/bin/python -m regime.report
Reads models/walkforward_results_{ours,paper,paper_log}.json and the matching
walkforward_predictions_*.csv (created by `python -m regime.train`), writes
docs/images/regime_*.png and docs/REGIME_RESULTS.md.
"""
from __future__ import annotations

import json
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import MODEL_DIR, ROOT, STRESS_PERIODS, VOL_FLOOR  # noqa: E402
from .metrics import daily_ic, evaluate, qlike  # noqa: E402

IMG = ROOT / "docs" / "images"
DOC = ROOT / "docs" / "REGIME_RESULTS.md"

# Validated categorical palette (dataviz reference, slots 1-4) + neutral for baselines.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
     "surface": "#fcfcfb"}

RUNS = {  # file suffix -> label
    "_ours": "RG-ResMoE (our settings, log inputs)",
    "_paper": "RG-ResMoE (authors' settings)",
    "_paper_log": "RG-ResMoE (authors' settings + log inputs)",
}
FLOOR = VOL_FLOOR  # applied to every model's forecasts before scoring
STRESS = {k: (a, b) for k, (_, a, b) in STRESS_PERIODS.items()}
SEGMENTS = {"all": "All assets", "stocks": "Stocks", "crypto": "Crypto",
            "covid_crash": "COVID crash (Feb–Apr 2020)", "2022_bear_market": "2022 bear market"}


def style(ax, title: str = "", ylabel: str = ""):
    ax.set_facecolor(C["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["axis"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    ax.grid(axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=C["ink"], fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=10)


def date_axis(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def fig(w=10, h=4.2):
    f, ax = plt.subplots(figsize=(w, h), dpi=150)
    f.patch.set_facecolor(C["surface"])
    return f, ax


def load() -> Dict[str, dict]:
    runs = {}
    for suffix, label in RUNS.items():
        rj = MODEL_DIR / f"walkforward_results{suffix}.json"
        pc = MODEL_DIR / f"walkforward_predictions{suffix}.csv"
        if rj.exists() and pc.exists():
            preds = pd.read_csv(pc, parse_dates=["date"]).set_index(["date", "symbol"])
            res = json.loads(rj.read_text())
            res["metrics"] = recompute(preds, res["config"]["seeds"])
            runs[suffix] = {"label": label, "res": res, "pred": preds}
    return runs


def recompute(pred: pd.DataFrame, seeds: List[int]) -> dict:
    """Metrics from saved forecasts with the volatility floor applied (raw QLIKE kept for reference)."""
    dates = pred.index.get_level_values("date")
    masks = {"all": None, "stocks": ~pred["is_crypto"], "crypto": pred["is_crypto"]}
    for name, (a, b) in STRESS.items():
        m = (dates >= a) & (dates <= b)
        if m.any():
            masks[name] = pd.Series(m, index=pred.index)
    out = {}
    for seg, mask in masks.items():
        d = pred if mask is None else pred[mask]
        seg_out = {}
        for key, prefix in (("rg_resmoe", "pred_moe"), ("plain_mlp", "pred_mlp"), ("base_only", "pred_base")):
            runs, raw = [], []
            for sd in seeds:
                col = f"{prefix}_s{sd}"
                raw.append(qlike(d["y"].to_numpy(), d[col].to_numpy()))
                runs.append(evaluate(d.assign(_p=np.maximum(d[col], FLOOR)), "_p"))
            agg = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                   for k in ("ic", "rmse", "r2", "qlike")}
            agg["qlike_raw"] = {"mean": float(np.mean(raw)), "std": float(np.std(raw))}
            agg["collapsed_seeds"] = f"{sum(r['ic'] < 0.1 for r in runs)}/{len(runs)}"
            seg_out[key] = agg
        seg_out["naive_rv20"] = evaluate(d.assign(_p=np.maximum(d["pred_naive"], FLOOR)), "_p")
        out[seg] = seg_out
    return out


def mean_seeds(pred: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in pred.columns if c.startswith(prefix + "_s")]
    return pred[cols].mean(axis=1)


# ---------------------------------------------------------------- figures
def fig_ic_by_window(runs: Dict[str, dict]) -> str:
    """Out-of-sample IC per 63-day test window."""
    f, ax = fig()
    series: List[tuple] = []
    main = runs.get("_paper_log") or next(iter(runs.values()))
    pred = main["pred"]
    win = main["res"]["windows"]
    starts = [pd.Timestamp(w["test_start"]) for w in win]
    ends = [pd.Timestamp(w["test_end"]) for w in win]

    def per_window(p: pd.DataFrame, col: pd.Series) -> List[float]:
        d = p.assign(_p=col)
        out = []
        for a, b in zip(starts, ends):
            sel = d[(d.index.get_level_values("date") >= a) & (d.index.get_level_values("date") <= b)]
            ic = daily_ic(sel, "_p", "y")
            out.append(ic.mean() if len(ic) else np.nan)
        return out

    colors = {"_ours": C["blue"], "_paper": C["orange"], "_paper_log": C["yellow"]}
    for suffix, r in runs.items():
        series.append((r["label"], per_window(r["pred"], mean_seeds(r["pred"], "pred_moe")), colors[suffix], "-"))
    ref = runs.get("_paper_log", main)
    series.append(("Plain MLP, same size (authors' + log)", per_window(ref["pred"], mean_seeds(ref["pred"], "pred_mlp")), C["aqua"], "-"))
    series.append(("Naive: last 20 days' volatility", per_window(pred, pred["pred_naive"]), C["muted"], "--"))

    for label, vals, color, ls in series:
        ax.plot(starts, vals, color=color, linewidth=2, linestyle=ls, marker="o", markersize=4,
                markeredgecolor=C["surface"], markeredgewidth=1, label=label)
    covid = pd.Timestamp("2020-02-18")
    ax.axvspan(covid, pd.Timestamp("2020-05-15"), color=C["grid"], alpha=0.6, zorder=0)
    ax.annotate("COVID crash", (covid, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                fontsize=8, color=C["ink2"])
    style(ax, "Out-of-sample IC per test window (higher is better)", "IC (rank correlation)")
    date_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, labelcolor=C["ink2"])
    path = IMG / "regime_ic_by_window.png"
    f.savefig(path, facecolor=C["surface"], bbox_inches="tight"); plt.close(f)
    return path.name


def fig_regime_timeline(run: dict) -> str:
    """Average gate weight per expert (stocks) over time, with market volatility below."""
    tl = run["res"]["regime_timeline"]
    d = pd.to_datetime(tl["date"])
    k = len([c for c in tl if c.startswith("pi")])
    W = np.vstack([np.asarray(tl[f"pi{i}"]) for i in range(k)])
    W = pd.DataFrame(W.T, index=d).rolling(5, min_periods=1).mean().T.to_numpy()  # 5-day smoothing
    colors = [C["blue"], C["orange"], C["aqua"], C["yellow"]][:k]
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), dpi=150, sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    f.patch.set_facecolor(C["surface"])
    ax1.stackplot(d, W, colors=colors, edgecolor=C["surface"], linewidth=0.6,
                  labels=[f"Expert {i}" for i in range(k)])
    ax1.set_ylim(0, 1); ax1.set_xlim(d.min(), d.max())
    style(ax1, "Regime gate: average expert weight across stocks", "Gate weight")
    ax1.text(0, 1.01, run["label"] + " · 5-day smoothing", transform=ax1.transAxes, fontsize=8.5, color=C["muted"])
    ax1.legend(frameon=False, fontsize=8.5, ncol=k, loc="upper left", bbox_to_anchor=(0, 1.0), labelcolor=C["ink2"])
    ax2.plot(d, np.asarray(tl["mkt_vol"]) * 100, color=C["ink2"], linewidth=1.5)
    style(ax2, "", "Market vol (%)")
    ax2.set_title("Market volatility (20-day, annualized)", loc="left", fontsize=10, color=C["ink2"])
    date_axis(ax2)
    path = IMG / "regime_gate_timeline.png"
    f.savefig(path, facecolor=C["surface"], bbox_inches="tight"); plt.close(f)
    return path.name


def fig_segments(runs: Dict[str, dict]) -> str:
    """IC by segment for each model (grouped bars)."""
    main = runs.get("_paper_log") or next(iter(runs.values()))
    segs = [s for s in SEGMENTS if s in main["res"]["metrics"]]
    models = []
    for suffix, label, color in (("_ours", "RG-ResMoE (our settings)", C["blue"]),
                                 ("_paper", "RG-ResMoE (authors')", C["orange"]),
                                 ("_paper_log", "RG-ResMoE (authors' + log)", C["yellow"])):
        if suffix in runs:
            models.append((label, runs[suffix], "rg_resmoe", color))
    ref = runs.get("_paper_log", main)
    models += [("Plain MLP (same size)", ref, "plain_mlp", C["aqua"]), ("Naive", ref, "naive_rv20", C["muted"])]
    f, ax = fig(10, 4.4)
    width = 0.8 / len(models)
    x = np.arange(len(segs))
    for i, (label, r, key, color) in enumerate(models):
        vals = []
        for s in segs:
            m = r["res"]["metrics"].get(s, {}).get(key)
            vals.append((m["ic"]["mean"] if isinstance(m.get("ic"), dict) else m.get("ic")) if m else np.nan)
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.92, color=color, label=label,
                      edgecolor=C["surface"], linewidth=1)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.5, color=C["ink2"])
    ax.set_xticks(x, [SEGMENTS[s] for s in segs], fontsize=9, color=C["ink2"])
    style(ax, "Out-of-sample IC by segment (higher is better)", "IC")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right", labelcolor=C["ink2"])
    ax.set_ylim(0, 0.85)
    path = IMG / "regime_ic_by_segment.png"
    f.tight_layout(); f.savefig(path, facecolor=C["surface"]); plt.close(f)
    return path.name


def fig_example(run: dict, symbol: str) -> str:
    """Forecast vs. realized 5-day volatility for one asset."""
    p = run["pred"]
    if symbol not in p.index.get_level_values("symbol"):
        symbol = p.index.get_level_values("symbol")[0]
    s = p.xs(symbol, level="symbol").sort_index()
    f, ax = fig(10, 3.8)
    ax.plot(s.index, s["y"] * 100, color=C["axis"], linewidth=1, label="Actual next-5-day volatility")
    ax.plot(s.index, s["pred_naive"] * 100, color=C["muted"], linewidth=1.2, linestyle="--", label="Naive forecast")
    ax.plot(s.index, mean_seeds(p, "pred_moe").xs(symbol, level="symbol").sort_index() * 100, color=C["blue"],
            linewidth=2, label="RG-ResMoE forecast")
    style(ax, f"{symbol}: forecast vs. actual volatility (out-of-sample)", "Annualized vol (%)")
    date_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=C["ink2"])
    path = IMG / f"regime_forecast_{symbol.replace('-', '_').lower()}.png"
    f.tight_layout(); f.savefig(path, facecolor=C["surface"]); plt.close(f)
    return path.name


# ---------------------------------------------------------------- tables
def fmt(m, key: str) -> str:
    if m is None or m.get(key) is None:
        return "–"
    v = m[key]
    return f"{v['mean']:.4f} ± {v['std']:.4f}" if isinstance(v, dict) else f"{v:.4f}"


def metrics_table(res: dict, seg: str) -> str:
    m = res["metrics"].get(seg)
    if not m:
        return ""
    rows = ["| Model | IC ↑ | RMSE ↓ | R² ↑ | QLIKE ↓ | QLIKE, no floor | Collapsed seeds |",
            "|---|---|---|---|---|---|---|"]
    names = {"rg_resmoe": "**RG-ResMoE**", "plain_mlp": "Plain MLP (same size)", "base_only": "Base network only",
             "naive_rv20": "Naive (last 20 days' vol)"}
    for key, name in names.items():
        r = m[key]
        rows.append(f"| {name} | {fmt(r, 'ic')} | {fmt(r, 'rmse')} | {fmt(r, 'r2')} | {fmt(r, 'qlike')} | "
                    f"{fmt(r, 'qlike_raw')} | {r.get('collapsed_seeds', '–')} |")
    return "\n".join(rows)


def comparison_table(runs: Dict[str, dict]) -> str:
    rows = ["| Run | Weights | IC ↑ | RMSE ↓ | QLIKE ↓ | IC, COVID crash ↑ | Training time |", "|---|---|---|---|---|---|---|"]
    for r in runs.values():
        res = r["res"]
        a, cv = res["metrics"]["all"]["rg_resmoe"], res["metrics"].get("covid_crash", {}).get("rg_resmoe")
        rows.append(f"| {r['label']} | {res['weights']['rg_resmoe']:,} | {fmt(a, 'ic')} | {fmt(a, 'rmse')} | "
                    f"{fmt(a, 'qlike')} | {fmt(cv, 'ic')} | {res['train_seconds'] / 60:.0f} min |")
    return "\n".join(rows)


def experts_table(res: dict) -> str:
    rows = ["| Expert | Dominant on | Average weight | Market vol when dominant | Realized vol when dominant |",
            "|---|---|---|---|---|"]
    for e in res["experts"]:
        rows.append(f"| {e['expert']} | {e['share_dominant']:.0%} of stock-days | {e['avg_weight']:.2f} | "
                    f"{(e['mean_market_vol'] or 0) * 100:.1f}% | {(e['mean_realized_vol'] or 0) * 100:.1f}% |")
    return "\n".join(rows)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    runs = load()
    if not runs:
        raise SystemExit("No results in models/. Run `python -m regime.train` first.")
    main_run = runs.get("_paper_log") or next(iter(runs.values()))
    best = runs.get("_paper_log") or runs.get("_paper", main_run)
    figs = {
        "ic": fig_ic_by_window(runs),
        "seg": fig_segments(runs),
        "gate": fig_regime_timeline(best),
        "nvda": fig_example(best, "NVDA"),
        "btc": fig_example(best, "BTC-USD"),
    }
    res = main_run["res"]
    lines = [
        "# Regime model results",
        "",
        "Auto-generated by `python -m regime.report` from `models/walkforward_results*.json`. "
        "Design and discussion: [REGIME_MODEL.md](REGIME_MODEL.md).",
        "",
        f"Out-of-sample period **{res['oos_period'][0]} to {res['oos_period'][1]}**, "
        f"{res['oos_rows']:,} forecasts per model, {len(res['windows'])} walk-forward windows, "
        f"{len(res['config']['seeds'])} seeds (mean ± std across seeds). Every forecast is for a period the model "
        "never saw during training. Forecasts are floored at 3% annualized volatility before scoring "
        "(see REGIME_MODEL.md); QLIKE without the floor is shown for reference.",
        "",
        "## Settings compared",
        "",
        comparison_table(runs),
        "",
        "## IC over time",
        "",
        f"![IC per window](images/{figs['ic']})",
        "",
        "## IC by segment",
        "",
        f"![IC by segment](images/{figs['seg']})",
        "",
    ]
    for suffix, r in runs.items():
        lines += [f"## Detailed metrics: {r['label']}", ""]
        for seg, name in SEGMENTS.items():
            t = metrics_table(r["res"], seg)
            if t:
                lines += [f"**{name}**", "", t, ""]
    lines += [
        f"## Regime gate ({best['label']})",
        "",
        f"![Gate timeline](images/{figs['gate']})",
        "",
        "When each expert dominates (stocks, first seed):",
        "",
        experts_table(best["res"]),
        "",
        "## Example forecasts",
        "",
        f"![NVDA](images/{figs['nvda']})",
        "",
        f"![BTC](images/{figs['btc']})",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(f"Wrote {DOC.relative_to(ROOT)} and {len(figs)} figures in docs/images/")


if __name__ == "__main__":
    main()
