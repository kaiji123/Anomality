"""Figures for docs/CRISIS_ROBUST_GATING.md: raw vs. log-scaled volatility inputs.

Usage: .venv/bin/python -m regime.figures_log
Needs models/walkforward_predictions_{paper,paper_log}.csv and walkforward_results_{paper,paper_log}.json.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import MODEL_DIR, VOL_FLOOR, TrainConfig  # noqa: E402
from .dataset import build_panel  # noqa: E402
from .metrics import daily_ic  # noqa: E402
from .report import C, IMG, date_axis, style  # noqa: E402

FLOOR = VOL_FLOOR
RAW, LOG = C["orange"], C["blue"]  # raw inputs = paper; log inputs = ours
DIV_POS, DIV_NEG = "#2a78d6", "#e34948"  # diverging pair (blue = better, red = worse)


def _load(sfx: str):
    p = pd.read_csv(MODEL_DIR / f"walkforward_predictions_{sfx}.csv", parse_dates=["date"]).set_index(["date", "symbol"])
    p["moe"] = p[[c for c in p if c.startswith("pred_moe_s")]].mean(axis=1).clip(lower=FLOOR)
    res = json.loads((MODEL_DIR / f"walkforward_results_{sfx}.json").read_text())
    return p, res


def _save(f, name: str) -> str:
    path = IMG / name
    f.savefig(path, facecolor=C["surface"], bbox_inches="tight")
    plt.close(f)
    return name


def fig_out_of_range() -> str:
    """Training-period market volatility vs. the COVID peak, on raw and log scales (in std devs)."""
    panel = build_panel(TrainConfig())
    mv = panel["mkt_vol"].groupby(level="date").first()
    train = mv[(mv.index >= "2018-02-01") & (mv.index < "2019-12-01")]
    peak_day = mv["2020-02-01":"2020-05-01"].idxmax()
    peak = mv[peak_day]
    f, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, sharey=True)
    f.patch.set_facecolor(C["surface"])
    maxes = []
    for ax, (title, values, peak_v, color) in zip(axes, [
        ("Raw inputs (paper)", train.to_numpy(), peak, RAW),
        ("Log inputs (ours)", np.log(train.to_numpy()), np.log(peak), LOG),
    ]):
        z = (values - values.mean()) / values.std()
        zp = (peak_v - values.mean()) / values.std()
        ax.hist(z, bins=30, color=color, edgecolor=C["surface"], linewidth=0.8)
        ax.axvline(zp, color=C["ink"], linewidth=2)
        right = zp < 8
        ax.annotate(f"COVID peak\n{peak:.0%} market vol\n= {zp:.1f} std devs", (zp, ax.get_ylim()[1] * 0.55),
                    xytext=(8 if right else -8, 0), textcoords="offset points", ha="left" if right else "right",
                    va="center", fontsize=8.5, color=C["ink"])
        ax.axvspan(z.min(), z.max(), color=C["grid"], alpha=0.5, zorder=0)
        maxes.append(z.max())
        ax.set_xlim(-3, 13)
        style(ax, title, "Days in training window" if ax is axes[0] else "")
        ax.set_xlabel("Std devs from the training average", color=C["ink2"], fontsize=9)
    f.suptitle("How far outside its training data the COVID crash pushed the regime gate's input",
               x=0.02, ha="left", fontsize=12, fontweight="bold", color=C["ink"], y=1.07)
    f.text(0.02, 0.985, f"Market volatility, Feb 2018 – Nov 2019 training window. Grey band = range seen in training "
           f"(up to {maxes[0]:.1f} std devs raw, {maxes[1]:.1f} log).", fontsize=8.5, color=C["muted"])
    return _save(f, "log_out_of_range.png")


def fig_gate_covid(res_raw: dict, res_log: dict) -> str:
    """Gate weights through the COVID crash, raw vs. log inputs (small multiples)."""
    colors = [C["blue"], C["orange"], C["aqua"], C["yellow"]]
    f, axes = plt.subplots(3, 1, figsize=(10, 6.4), dpi=150, sharex=True,
                           gridspec_kw={"height_ratios": [2, 2, 1], "hspace": 0.35})
    f.patch.set_facecolor(C["surface"])
    for ax, (title, res) in zip(axes[:2], [("Raw inputs (paper)", res_raw), ("Log inputs (ours)", res_log)]):
        tl = pd.DataFrame(res["regime_timeline"]).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")
        tl = tl["2020-01-02":"2020-07-31"]
        k = len([c for c in tl if c.startswith("pi")])
        ax.stackplot(tl.index, [tl[f"pi{i}"] for i in range(k)], colors=colors[:k], edgecolor=C["surface"],
                     linewidth=0.5, labels=[f"Expert {i}" for i in range(k)])
        ax.set_ylim(0, 1)
        ax.set_xlim(tl.index.min(), tl.index.max())
        style(ax, title, "Gate weight")
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1, 1.02), labelcolor=C["ink2"])
    tl = pd.DataFrame(res_log["regime_timeline"]).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")
    tl = tl["2020-01-02":"2020-07-31"]
    axes[2].plot(tl.index, tl["mkt_vol"] * 100, color=C["ink2"], linewidth=1.5)
    style(axes[2], "Market volatility (%)", "")
    date_axis(axes[2])
    f.suptitle("Regime gate through the COVID crash (average across stocks, out-of-sample)",
               x=0.02, ha="left", fontsize=12, fontweight="bold", color=C["ink"], y=1.0)
    return _save(f, "log_gate_covid.png")


def fig_window_gains(raw: pd.DataFrame, log: pd.DataFrame, res_log: dict) -> tuple:
    """IC improvement from log inputs in each walk-forward test window."""
    d = (daily_ic(log, "moe") - daily_ic(raw, "moe")).dropna()
    starts, gains = [], []
    for w in res_log["windows"]:
        a, b = pd.Timestamp(w["test_start"]), pd.Timestamp(w["test_end"])
        starts.append(a)
        gains.append(d[(d.index >= a) & (d.index <= b)].mean())
    gains = np.array(gains)
    f, ax = plt.subplots(figsize=(10, 3.8), dpi=150)
    f.patch.set_facecolor(C["surface"])
    ax.bar(starts, gains, width=50, color=[DIV_POS if g > 0 else DIV_NEG for g in gains],
           edgecolor=C["surface"], linewidth=1)
    ax.axhline(0, color=C["axis"], linewidth=1)
    covid = pd.Timestamp("2020-02-18")
    i = int(np.argmin([abs((s - covid).days) for s in starts]))
    ax.annotate("COVID crash", (starts[i], gains[i]), xytext=(0, 6), textcoords="offset points", ha="center",
                fontsize=8, color=C["ink2"])
    n_better = int((gains > 0).sum())
    style(ax, f"Log inputs improved IC in {n_better} of {len(gains)} test windows", "IC gain vs. raw inputs")
    ax.text(0, 1.01, "Blue = log inputs better, red = worse. RG-ResMoE, authors' settings, mean of 3 seeds.",
            transform=ax.transAxes, fontsize=8.5, color=C["muted"])
    date_axis(ax)
    return _save(f, "log_window_gains.png"), n_better, len(gains)


def fig_covid_forecast(raw: pd.DataFrame, log: pd.DataFrame) -> str:
    """Average forecast vs. actual volatility across stocks through the crash."""
    def avg(p: pd.DataFrame, col: str) -> pd.Series:
        s = p[~p["is_crypto"]]
        return s[col].groupby(level="date").mean()["2020-01-02":"2020-07-31"] * 100
    f, ax = plt.subplots(figsize=(10, 3.8), dpi=150)
    f.patch.set_facecolor(C["surface"])
    ax.plot(avg(raw, "y").index, avg(raw, "y"), color=C["ink"], linewidth=1.2, label="Actual next-5-day volatility")
    ax.plot(avg(raw, "moe").index, avg(raw, "moe"), color=RAW, linewidth=2, label="Forecast, raw inputs (paper)")
    ax.plot(avg(log, "moe").index, avg(log, "moe"), color=LOG, linewidth=2, label="Forecast, log inputs (ours)")
    ax.plot(avg(raw, "pred_naive").index, avg(raw, "pred_naive"), color=C["muted"], linewidth=1.2, linestyle="--",
            label="Naive forecast")
    style(ax, "Through the COVID crash: forecast vs. actual (average across stocks)", "Annualized vol (%)")
    date_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=C["ink2"])
    return _save(f, "log_covid_forecast.png")


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    raw, res_raw = _load("paper")
    log, res_log = _load("paper_log")
    names = [fig_out_of_range(), fig_gate_covid(res_raw, res_log)]
    name, n_better, n = fig_window_gains(raw, log, res_log)
    names += [name, fig_covid_forecast(raw, log)]
    print(f"Wrote {', '.join(names)} to docs/images/ ({n_better}/{n} windows improved)")


if __name__ == "__main__":
    main()
