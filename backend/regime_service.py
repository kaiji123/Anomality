"""Serves the trained RG-ResMoE regime model to the app.

Two versions are compared side by side:
  log  = authors' settings + log-scaled volatility inputs (ours, recommended) -> models/rg_resmoe.pt
  raw  = authors' exact settings, raw inputs (the paper's design)            -> models/rg_resmoe_paper.pt

History comes from the walk-forward test predictions (out-of-sample only). Today's forecast
comes from the final models, trained on the most recent 504 days.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from regime.config import MODEL_DIR, X_COLS, Z_COLS, TrainConfig
from regime.dataset import build_panel
from regime.model import build_moe
from regime.train import Scaler

FLOOR = 0.03  # forecasts below 3% annualized vol are clamped
SURPRISE = 2.0  # realized vol at least 2x the forecast = trained anomaly
STRESS_PERIODS = {"COVID crash": ("2020-02-19", "2020-04-30"), "2022 bear market": ("2022-01-03", "2022-10-31")}
REGIME_NAMES = ["Calm", "Normal", "Elevated", "Stress"]  # experts ranked by market vol when dominant

_lock = threading.Lock()
_state: Dict[str, object] = {}


class RegimeUnavailable(Exception):
    pass


def _load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in TrainConfig.__dataclass_fields__})
    model = build_moe(len(X_COLS), len(Z_COLS), cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    sd = ck["scaler"]
    sc = Scaler(**{k: (np.asarray(v) if isinstance(v, list) else v) for k, v in sd.items()})
    return {"model": model, "scaler": sc, "meta": ck["meta"]}


SERVING_COLS = ["date", "symbol", "y", "rv20", "mkt_vol", "is_crypto", "pred_naive", "pred", "pi0", "pi1", "pi2", "pi3"]


def _load_preds(name: str) -> Optional[pd.DataFrame]:
    """Walk-forward test forecasts. Uses the full training output if present, otherwise the
    compact copy shipped in git (models/serving_<name>.csv.gz)."""
    full = MODEL_DIR / f"walkforward_predictions_{name}.csv"
    compact = MODEL_DIR / f"serving_{name}.csv.gz"
    if full.exists():
        p = pd.read_csv(full, parse_dates=["date"])
        seeds = [c for c in p.columns if c.startswith("pred_moe_s")]
        p["pred"] = p[seeds].mean(axis=1).clip(lower=FLOOR)
    elif compact.exists():
        p = pd.read_csv(compact, parse_dates=["date"])
    else:
        return None
    pis = [c for c in p.columns if c.startswith("pi")]
    p["top_w"] = p[pis].max(axis=1)  # gate concentration; doesn't depend on expert order
    return p.set_index(["symbol", "date"]).sort_index()


def export_serving() -> None:
    """Write the compact forecast files the app needs (run after training)."""
    for name in ("paper_log", "paper"):
        p = _load_preds(name)
        if p is not None:
            out = p.reset_index()[SERVING_COLS]
            out.to_csv(MODEL_DIR / f"serving_{name}.csv.gz", index=False, float_format="%.4f", compression="gzip")


@torch.no_grad()
def _predict(m: dict, df: pd.DataFrame):
    x, z, _ = m["scaler"].tensors(df, "cpu")
    yhat, _, _, pi = m["model"](x, z)
    return np.maximum(m["scaler"].unscale(yhat.numpy()), FLOOR), pi.numpy()


def _expert_names(m: dict, panel: pd.DataFrame) -> list:
    """Name experts by how volatile the market is when each one dominates (final model, its own window)."""
    start, end = m["meta"]["trained_from"], m["meta"]["trained_through"]
    d = panel.index.get_level_values("date")
    win = panel[(d >= start) & (d <= end) & ~panel["is_crypto"]]
    _, pi = _predict(m, win)
    dom = pi.argmax(1)
    k = pi.shape[1]
    vol = [win["mkt_vol"].to_numpy()[dom == i].mean() if (dom == i).any() else np.nan for i in range(k)]
    order = np.argsort(np.nan_to_num(vol, nan=np.nanmean(vol)))
    names = [""] * k
    for rank, i in enumerate(order):
        names[i] = REGIME_NAMES[min(rank, len(REGIME_NAMES) - 1)]
    return names


def _ensure_loaded() -> dict:
    with _lock:
        if _state.get("loaded_at") and time.time() - _state["loaded_at"] < 600:
            return _state
        if not (MODEL_DIR / "rg_resmoe.pt").exists() or not (MODEL_DIR / "rg_resmoe_paper.pt").exists():
            raise RegimeUnavailable("Trained models not found. Run `python -m regime.train` and "
                                    "`python -m regime.train --preset paper` first.")
        panel = build_panel(TrainConfig())
        log_m, raw_m = _load_model(MODEL_DIR / "rg_resmoe.pt"), _load_model(MODEL_DIR / "rg_resmoe_paper.pt")
        _state.update({
            "panel": panel,
            "log": log_m,
            "raw": raw_m,
            "names": _expert_names(log_m, panel),
            "preds_log": _load_preds("paper_log"),
            "preds_raw": _load_preds("paper"),
            "market": None,
            "track": None,
            "loaded_at": time.time(),
        })
        return _state


def _gate_input_z(m: dict, mkt_vol: float) -> float:
    """How many std devs today's market vol is from the final model's training average, as the gate sees it."""
    sc = m["scaler"]
    v = np.log(max(mkt_vol, 1e-4)) if sc.log_vol else mkt_vol
    i = Z_COLS.index("mkt_vol")
    return float((v - sc.z_mean[i]) / sc.z_std[i])


def _metrics(y: np.ndarray, f: np.ndarray) -> dict:
    ratio = np.maximum(y, 1e-4) ** 2 / np.maximum(f, 1e-3) ** 2
    return {"rmse": round(float(np.sqrt(np.mean((y - f) ** 2))), 4),
            "qlike": round(float(np.mean(ratio - np.log(ratio) - 1)), 4)}


def regime_for(symbol: str) -> dict:
    st = _ensure_loaded()
    symbol = symbol.upper()
    panel: pd.DataFrame = st["panel"]
    if symbol not in panel.index.get_level_values("symbol"):
        raise RegimeUnavailable(f"{symbol} isn't in the regime model's 56-asset training universe yet.")
    rows = panel.xs(symbol, level="symbol", drop_level=False).sort_index()
    latest = rows.iloc[[-1]]
    f_log, pi_log = _predict(st["log"], latest)
    f_raw, pi_raw = _predict(st["raw"], latest)
    mkt_vol = float(latest["mkt_vol"].iloc[0])
    names = st["names"]
    weights = pi_log[0]
    out = {
        "symbol": symbol,
        "latest": {
            "date": str(latest.index.get_level_values("date")[0].date()),
            "forecast_log": round(float(f_log[0]), 4),
            "forecast_raw": round(float(f_raw[0]), 4),
            "realized_20d": round(float(latest["rv20"].iloc[0]), 4),
            "market_vol": round(mkt_vol, 4),
            "gate_input_z": {"raw": round(_gate_input_z(st["raw"], mkt_vol), 2),
                             "log": round(_gate_input_z(st["log"], mkt_vol), 2)},
            "weights": [{"expert": i, "name": names[i], "weight": round(float(w), 4)} for i, w in enumerate(weights)],
            "regime": names[int(weights.argmax())],
            "top_weight": {"log": round(float(pi_log[0].max()), 4), "raw": round(float(pi_raw[0].max()), 4)},
        },
        "model": {
            "trained_from": st["log"]["meta"]["trained_from"],
            "trained_through": st["log"]["meta"]["trained_through"],
            "weights": st["log"]["meta"]["weights"],
        },
        "history": None,
    }

    pl, pr = st["preds_log"], st["preds_raw"]
    if pl is not None and pr is not None and symbol in pl.index.get_level_values("symbol"):
        a = pl.xs(symbol, level="symbol")
        b = pr.xs(symbol, level="symbol").reindex(a.index)
        surprise = (a["y"] / a["pred"]).to_numpy()
        out["history"] = {
            "date": [str(d.date()) for d in a.index],
            "actual": a["y"].round(4).tolist(),
            "pred_log": a["pred"].round(4).tolist(),
            "pred_raw": b["pred"].round(4).tolist(),
            "naive": a["pred_naive"].round(4).tolist(),
            "top_w_log": a["top_w"].round(4).tolist(),
            "top_w_raw": b["top_w"].round(4).tolist(),
            "market_vol": a["mkt_vol"].round(4).tolist(),
            "surprises": [
                {"date": str(d.date()), "actual": round(float(y), 4), "forecast": round(float(f), 4),
                 "ratio": round(float(r), 2)}
                for d, y, f, r in zip(a.index, a["y"], a["pred"], surprise) if r >= SURPRISE
            ][::-1],
        }
        y = a["y"].to_numpy()
        mets = {"log": _metrics(y, a["pred"].to_numpy()), "raw": _metrics(y, b["pred"].to_numpy()),
                "naive": _metrics(y, np.maximum(a["pred_naive"].to_numpy(), FLOOR))}
        crises = {}
        for name, (s, e) in STRESS_PERIODS.items():
            m = (a.index >= s) & (a.index <= e)
            if m.sum() >= 5:
                crises[name] = {"log": _metrics(y[m], a["pred"].to_numpy()[m]),
                                "raw": _metrics(y[m], b["pred"].to_numpy()[m]),
                                "naive": _metrics(y[m], np.maximum(a["pred_naive"].to_numpy()[m], FLOOR)),
                                "top_w_log": round(float(a["top_w"][m].mean()), 3),
                                "top_w_raw": round(float(b["top_w"][m].mean()), 3)}
        out["metrics"] = {"all": mets, "crises": crises}
    return out


def _agreement(pi: np.ndarray) -> float:
    """Share of assets whose dominant expert is the most common one."""
    dom = pi.argmax(1)
    return float(np.bincount(dom, minlength=pi.shape[1]).max() / len(dom)) if len(dom) else float("nan")


def _timeline(preds: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if preds is None:
        return None
    s = preds[~preds["is_crypto"]].reset_index()
    pis = [c for c in s.columns if c.startswith("pi")]
    dom = s[pis].to_numpy().argmax(1)
    s = s.assign(dom=dom)
    g = s.groupby("date")
    return pd.DataFrame({
        "agree": g["dom"].agg(lambda x: x.value_counts(normalize=True).iloc[0]),
        "mkt_vol": g["mkt_vol"].first(),
    })


def market_overview() -> dict:
    st = _ensure_loaded()
    with _lock:
        cached = st.get("market")
    if cached:
        return cached
    panel: pd.DataFrame = st["panel"]
    last = panel.index.get_level_values("date").max()
    today = panel.xs(last, level="date", drop_level=False)
    f_log, pi_log = _predict(st["log"], today)
    f_raw, pi_raw = _predict(st["raw"], today)
    stocks = ~today["is_crypto"].to_numpy()
    names = st["names"]
    avg_w = pi_log[stocks].mean(0)
    mkt_vol = float(today["mkt_vol"].iloc[0])
    covid_peak = 1.02  # market vol on 27 Mar 2020
    out = {
        "date": str(last.date()),
        "regime": names[int(avg_w.argmax())],
        "weights": [{"expert": i, "name": names[i], "weight": round(float(w), 4)} for i, w in enumerate(avg_w)],
        "market_vol": round(mkt_vol, 4),
        "agreement": {"log": round(_agreement(pi_log[stocks]), 3), "raw": round(_agreement(pi_raw[stocks]), 3)},
        "gate_input_z": {"raw": round(_gate_input_z(st["raw"], mkt_vol), 2), "log": round(_gate_input_z(st["log"], mkt_vol), 2)},
        "covid_peak_z": {"raw": round(_gate_input_z(st["raw"], covid_peak), 1), "log": round(_gate_input_z(st["log"], covid_peak), 1),
                         "market_vol": covid_peak},
        "assets": {
            sym: {"forecast": round(float(fl), 4), "forecast_raw": round(float(fr), 4),
                  "realized_20d": round(float(rv), 4), "regime": names[int(p.argmax())]}
            for sym, fl, fr, rv, p in zip(today.index.get_level_values("symbol"), f_log, f_raw, today["rv20"], pi_log)
        },
        "model": {"trained_through": st["log"]["meta"]["trained_through"], "weights": st["log"]["meta"]["weights"]},
        "history": None,
    }
    tl_log, tl_raw = _timeline(st["preds_log"]), _timeline(st["preds_raw"])
    if tl_log is not None and tl_raw is not None:
        tl_raw = tl_raw.reindex(tl_log.index)
        out["history"] = {
            "date": [str(d.date()) for d in tl_log.index],
            "agree_log": tl_log["agree"].round(3).tolist(),
            "agree_raw": tl_raw["agree"].round(3).tolist(),
            "market_vol": tl_log["mkt_vol"].round(4).tolist(),
        }
        crisis = {}
        for name, (s, e) in STRESS_PERIODS.items():
            m = (tl_log.index >= s) & (tl_log.index <= e)
            crisis[name] = {"agree_log": round(float(tl_log["agree"][m].mean()), 3),
                            "agree_raw": round(float(tl_raw["agree"][m].mean()), 3)}
        calm = (tl_log.index >= "2019-06-01") & (tl_log.index <= "2020-01-31")
        crisis["Calm 2019"] = {"agree_log": round(float(tl_log["agree"][calm].mean()), 3),
                               "agree_raw": round(float(tl_raw["agree"][calm].mean()), 3)}
        out["crisis_agreement"] = crisis
    with _lock:
        st["market"] = out
    return out


# ---------------------------------------------------------------- plain-language view for users
WEEK = float(np.sqrt(5 / 252))  # annualized vol -> typical one-week move (1 std dev)
MOODS = [(0.35, "Calm", "Quieter than usual. Prices tend to move in small steps."),
         (0.70, "Normal", "A typical week, with ordinary ups and downs."),
         (0.90, "Choppy", "Busier than usual. Bigger daily swings, and news moves prices more."),
         (1.01, "Stormy", "Turbulent. Large swings in both directions are likely, like in a crash.")]


def weekly(v: float) -> float:
    return round(float(v) * WEEK * 100, 2)  # in percent


def mood_for(pctile: float, history: Optional[pd.Series] = None, value: Optional[float] = None) -> dict:
    """Mood from where `value` sits in `history` (annualized vols). With history, also returns the
    weekly-move range of each mood band so users can see the scale."""
    out = None
    for cut, name, text in MOODS:
        if pctile < cut:
            out = {"name": name, "text": text, "percentile": round(float(pctile), 3)}
            break
    if out is None:
        out = {"name": MOODS[-1][1], "text": MOODS[-1][2], "percentile": round(float(pctile), 3)}
    if history is not None and len(history):
        cuts = [weekly(history.quantile(c)) for c, _, _ in MOODS[:-1]]
        out["scale"] = {
            "names": [n for _, n, _ in MOODS],
            "cuts": cuts,  # weekly-move boundaries between Calm|Normal|Choppy|Stormy
            "low": weekly(history.quantile(0.02)),
            "high": weekly(max(history.quantile(0.98), value or 0)),
            "value": weekly(value) if value is not None else None,
            "since": str(history.index.min().date()) if hasattr(history.index, "min") else None,
        }
    return out


def _research(a: pd.DataFrame, b: pd.DataFrame, res: dict) -> dict:
    """Log vs. raw inputs: IC, windows improved, significance (daily IC differences)."""
    from scipy import stats

    from regime.metrics import daily_ic
    ia = daily_ic(a.set_index(["date", "symbol"]), "pred")
    ib = daily_ic(b.set_index(["date", "symbol"]), "pred")
    d = (ia - ib).dropna()
    better = 0
    for w in res["windows"]:
        m = (d.index >= w["test_start"]) & (d.index <= w["test_end"])
        better += d[m].mean() > 0
    wk = d.groupby(pd.Grouper(freq="W")).mean().dropna()
    return {"ic_log": round(float(ia.mean()), 4), "ic_raw": round(float(ib.mean()), 4),
            "windows_better": int(better), "windows": len(res["windows"]),
            "p_value": float(stats.ttest_1samp(wk, 0).pvalue), "days_better": round(float((d > 0).mean()), 3)}


def track_record() -> Optional[dict]:
    """Plain-language track record over all out-of-sample forecasts (all assets)."""
    st = _ensure_loaded()
    with _lock:
        if st.get("track"):
            return st["track"]
    pl, pr = st["preds_log"], st["preds_raw"]
    if pl is None or pr is None:
        return None
    a = pl.reset_index()
    b = pr.reindex(pl.index).reset_index()
    naive = a["pred_naive"].clip(lower=FLOOR)
    miss = lambda f: (np.abs(f - a["y"]) * WEEK * 100)
    m_ours, m_orig, m_simple = miss(a["pred"]), miss(b["pred"]), miss(naive)
    res = __import__("json").loads((MODEL_DIR / "walkforward_results_paper_log.json").read_text())
    wins_simple = wins_orig = 0
    for w in res["windows"]:
        m = (a["date"] >= w["test_start"]) & (a["date"] <= w["test_end"])
        wins_simple += m_ours[m].mean() < m_simple[m].mean()
        wins_orig += m_ours[m].mean() < m_orig[m].mean()
    cov = (a["date"] >= "2020-02-19") & (a["date"] <= "2020-04-30")
    rk = a.groupby("date")["pred"].rank(pct=True)
    out = {
        "periods": len(res["windows"]),
        "since": str(a["date"].min().date()),
        "beats_simple": int(wins_simple),
        "beats_original": int(wins_orig),
        "miss": {"ours": round(float(m_ours.mean()), 2), "simple": round(float(m_simple.mean()), 2),
                 "original": round(float(m_orig.mean()), 2)},
        "covid_miss": {"ours": round(float(m_ours[cov].mean()), 2), "simple": round(float(m_simple[cov].mean()), 2),
                       "original": round(float(m_orig[cov].mean()), 2)},
        "ranking": {"top": weekly(a["y"][rk >= 0.8].mean()), "bottom": weekly(a["y"][rk <= 0.2].mean())},
        "calmer_hit_rate": round(float((a["y"] < a["rv20"])[a["pred"] < 0.9 * a["rv20"]].mean()), 3),
        "research": _research(a, b, res),
    }
    with _lock:
        st["track"] = out
    return out


def plain_market() -> dict:
    m = market_overview()
    st = _ensure_loaded()
    assets = m["assets"]
    crypto = set(st["panel"][st["panel"]["is_crypto"]].index.get_level_values("symbol"))
    stock_fc = [v["forecast"] for s, v in assets.items() if s not in crypto]
    crypto_fc = [v["forecast"] for s, v in assets.items() if s in crypto]
    avg_now = float(np.mean(stock_fc))
    out = {"date": m["date"], "model": m["model"], "track": track_record(), "technical": m}
    pl = st["preds_log"]
    if pl is not None:
        hist = pl[~pl["is_crypto"]].groupby(level="date")[["pred", "y"]].mean()
        pct = float((hist["pred"] < avg_now).mean())
        mood_hist = hist["pred"]
        out["history"] = {"date": [str(d.date()) for d in hist.index],
                          "forecast": [weekly(v) for v in hist["pred"]],
                          "actual": [weekly(v) for v in hist["y"]]}
        out["usual_stock_move"] = weekly(hist["pred"].median())
    else:
        pct, mood_hist = 0.5, None
    out["mood"] = mood_for(pct, mood_hist, avg_now)
    if pl is not None:
        out["mood"]["levels"] = mood_levels(hist.dropna())
    out["stock_move"] = weekly(avg_now)
    out["crypto_move"] = weekly(np.mean(crypto_fc)) if crypto_fc else None
    if pl is not None:
        for sym, v in assets.items():
            if sym in pl.index.get_level_values("symbol"):
                hist_sym = pl.xs(sym, level="symbol")["pred"]
                v["mood"] = mood_for(float((hist_sym < v["forecast"]).mean()))["name"]
    ranked = sorted(assets.items(), key=lambda kv: -kv[1]["forecast"])
    out["biggest"] = [{"symbol": s, "move": weekly(v["forecast"]), "crypto": s in crypto} for s, v in ranked[:5]]
    out["biggest_stocks"] = [{"symbol": s, "move": weekly(v["forecast"])} for s, v in ranked if s not in crypto][:3]
    out["calmest"] = [{"symbol": s, "move": weekly(v["forecast"])} for s, v in ranked[::-1] if s not in crypto][:3]
    return out


def _events(surprises: list) -> list:
    """Merge surprise days less than a week apart into one episode (keep the biggest day)."""
    days = sorted(surprises, key=lambda s: s["date"])
    events, cur = [], None
    for s in days:
        d = pd.Timestamp(s["date"])
        if cur and (d - cur["_last"]).days <= 7:
            cur["_last"] = d
            cur["days"] += 1
            if s["ratio"] > cur["ratio"]:
                cur.update({"peak": s["date"], "ratio": s["ratio"], "actual": s["actual"], "forecast": s["forecast"]})
        else:
            cur = {"start": s["date"], "peak": s["date"], "ratio": s["ratio"], "actual": s["actual"],
                   "forecast": s["forecast"], "days": 1, "_last": d}
            events.append(cur)
    return [{**{k: v for k, v in e.items() if k != "_last"}, "actual": weekly(e["actual"]),
             "forecast": weekly(e["forecast"])} for e in events][::-1]


def mood_levels(hist: pd.DataFrame) -> list:
    """For each mood: share of weeks, what actually happened, and months it was the dominant mood.
    `hist` has a forecast column `pred` and realized column `y` (annualized), indexed by date."""
    cuts = [hist["pred"].quantile(c) for c, _, _ in MOODS[:-1]]
    labels = pd.cut(hist["pred"], [-np.inf, *cuts, np.inf], labels=[n for _, n, _ in MOODS])
    monthly = labels.groupby(hist.index.to_period("M")).agg(lambda x: x.value_counts().idxmax())
    out = []
    for _, name, text in MOODS:
        sel = hist[labels == name]
        months = [str(m) for m in monthly[monthly == name].index]
        out.append({"name": name, "text": text, "share": round(float(len(sel) / len(hist)), 3),
                    "actual_avg": weekly(sel["y"].mean()) if len(sel) else None,
                    "actual_rough": weekly(sel["y"].quantile(0.9)) if len(sel) else None,
                    "months": months})
    return out


def plain_asset(symbol: str) -> dict:
    r = regime_for(symbol)
    st = _ensure_loaded()
    L = r["latest"]
    out = {"symbol": r["symbol"], "date": L["date"], "move": weekly(L["forecast_log"]),
           "move_original": weekly(L["forecast_raw"]), "last_month_move": weekly(L["realized_20d"]),
           "technical": r}
    ratio = L["forecast_log"] / max(L["realized_20d"], 1e-6)
    out["trend"] = "calmer" if ratio < 0.9 else "wilder" if ratio > 1.1 else "steady"
    pl = st["preds_log"]
    if pl is not None and r["symbol"] in pl.index.get_level_values("symbol"):
        a = pl.xs(r["symbol"], level="symbol")
        out["usual_move"] = weekly(a["pred"].median())
        out["mood"] = mood_for(float((a["pred"] < L["forecast_log"]).mean()), a["pred"], L["forecast_log"])
        out["mood"]["levels"] = mood_levels(a[["pred", "y"]].dropna())
        h = r["history"]
        y = np.asarray(h["actual"], dtype=float)
        miss = lambda f: float(np.nanmean(np.abs(np.asarray(f, dtype=float) - y)) * WEEK * 100)
        dates = np.array(h["date"])
        cov = (dates >= "2020-02-19") & (dates <= "2020-04-30")
        missc = lambda f: float(np.nanmean(np.abs(np.asarray(f, dtype=float)[cov] - y[cov])) * WEEK * 100) if cov.any() else None
        naive = np.maximum(np.asarray(h["naive"], dtype=float), FLOOR)
        out["record"] = {"ours": round(miss(h["pred_log"]), 2), "simple": round(miss(naive), 2),
                         "original": round(miss(h["pred_raw"]), 2),
                         "covid": None if not cov.any() else {"ours": round(missc(h["pred_log"]), 2),
                                                               "simple": round(missc(naive), 2),
                                                               "original": round(missc(h["pred_raw"]), 2)}}
        out["history"] = {"date": h["date"], "actual": [weekly(v) for v in h["actual"]],
                          "forecast": [weekly(v) for v in h["pred_log"]],
                          "original": [weekly(v) for v in h["pred_raw"]],
                          "surprises": [{**s, "actual": weekly(s["actual"]), "forecast": weekly(s["forecast"])}
                                        for s in h["surprises"]],
                          "events": _events(h["surprises"])}
    else:
        out["mood"] = None
    return out


if __name__ == "__main__":  # python -m backend.regime_service  -> refresh the compact files after training
    export_serving()
    print("Wrote models/serving_paper_log.csv.gz and models/serving_paper.csv.gz")
