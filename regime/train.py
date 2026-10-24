"""Train and evaluate RG-ResMoE with walk-forward testing.

Usage:
    .venv/bin/python -m regime.train                    # recommended settings, all windows, 3 seeds (~12 min)
    .venv/bin/python -m regime.train --quick            # last 4 windows, 1 seed (smoke test)
    .venv/bin/python -m regime.train --preset paper     # authors' exact settings (raw inputs)
    .venv/bin/python -m regime.train --preset ours      # our original settings (width 64)
    .venv/bin/python -m regime.train --help

Presets: "recommended" = authors' settings + log-scaled volatility inputs (best in our tests).

Outputs (in models/), with the preset name as suffix:
    rg_resmoe_<preset>.pt                    final model trained on the most recent 504 days
    walkforward_results_<preset>.json        out-of-sample metrics vs. baselines, regime summary
    walkforward_predictions_<preset>.csv     every out-of-sample forecast
    rg_resmoe.pt                             copy of the recommended model (what the app loads)
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import LOG_COLS, MODEL_DIR, TARGET, X_COLS, Z_COLS, TrainConfig, paper_config
from .dataset import build_panel
from .metrics import evaluate
from .model import PlainMLP, RGResMoE, build_moe, matched_width, n_params

STRESS_PERIODS = {
    "covid_crash": ("2020-02-19", "2020-04-30"),
    "2022_bear_market": ("2022-01-03", "2022-10-31"),
}


# ---------------------------------------------------------------- data prep
@dataclass
class Scaler:
    """Standardisation fitted on training rows only."""
    x_mean: np.ndarray
    x_std: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    y_mean: float
    y_std: float
    log_vol: bool = False

    @staticmethod
    def _inputs(df: pd.DataFrame, log_vol: bool):
        if log_vol:
            df = df.copy()
            df[LOG_COLS] = np.log(df[LOG_COLS].clip(lower=1e-4))
        return df[X_COLS].to_numpy(), df[Z_COLS].to_numpy()

    @classmethod
    def fit(cls, df: pd.DataFrame, log_vol: bool = False, scale_target: bool = True) -> "Scaler":
        x, z = cls._inputs(df, log_vol)
        y = df[TARGET].to_numpy()
        y_mean, y_std = (float(y.mean()), float(y.std() + 1e-8)) if scale_target else (0.0, 1.0)
        return cls(x.mean(0), x.std(0) + 1e-8, z.mean(0), z.std(0) + 1e-8, y_mean, y_std, log_vol)

    def tensors(self, df: pd.DataFrame, device: str):
        xr, zr = self._inputs(df, self.log_vol)
        x = torch.tensor((xr - self.x_mean) / self.x_std, dtype=torch.float32, device=device)
        z = torch.tensor((zr - self.z_mean) / self.z_std, dtype=torch.float32, device=device)
        y = torch.tensor((df[TARGET].fillna(0).to_numpy() - self.y_mean) / self.y_std, dtype=torch.float32, device=device)
        return x.clamp(-10, 10), z.clamp(-10, 10), y

    def unscale(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------- training
def fit(params, train_loss: Callable[[], torch.Tensor], val_loss: Callable[[], float],
        module: torch.nn.Module, cfg: TrainConfig, lr: float = 0.0) -> Tuple[int, float]:
    """Full-batch Adam with early stopping on validation loss; restores the best weights."""
    opt = torch.optim.Adam(params, lr=lr or cfg.lr)
    best, best_state, bad, epoch = float("inf"), copy.deepcopy(module.state_dict()), 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        module.train()
        opt.zero_grad()
        loss = train_loss()
        loss.backward()
        opt.step()
        if epoch % cfg.check_every:
            continue
        module.eval()
        with torch.no_grad():
            v = val_loss()
        if v < best - 1e-7:
            best, best_state, bad = v, copy.deepcopy(module.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    module.load_state_dict(best_state)
    return epoch, best


def train_rg_resmoe(train: pd.DataFrame, val: pd.DataFrame, cfg: TrainConfig, seed: int):
    torch.manual_seed(seed)
    sc = Scaler.fit(train, cfg.log_vol_inputs, cfg.scale_target)
    xt, zt, yt = sc.tensors(train, cfg.device)
    xv, zv, yv = sc.tensors(val, cfg.device)
    model = build_moe(len(X_COLS), len(Z_COLS), cfg).to(cfg.device)
    mse = torch.nn.functional.mse_loss

    # Stage 1: base network only.
    e1, _ = fit(model.base.parameters(),
                lambda: mse(model.base(xt).squeeze(-1), yt),
                lambda: mse(model.base(xv).squeeze(-1), yv).item(), model, cfg)
    for p in model.base.parameters():
        p.requires_grad_(False)

    # Stage 2: experts + gate, base frozen.
    K = cfg.experts

    def stage2_loss():
        yhat, _, corr, pi = model(xt, zt)
        load = pi.mean(0)
        return mse(yhat, yt) + cfg.alpha * (corr ** 2).mean() + cfg.lambda_lb * ((load - 1.0 / K) ** 2).sum()

    e2, _ = fit(model.stage2_parameters(), stage2_loss, lambda: mse(model(xv, zv)[0], yv).item(), model, cfg,
                lr=cfg.lr_experts or cfg.lr)
    model.eval()
    return model, sc, {"epochs_stage1": e1, "epochs_stage2": e2}


def train_mlp(train: pd.DataFrame, val: pd.DataFrame, cfg: TrainConfig, seed: int, width: int):
    torch.manual_seed(seed)
    sc = Scaler.fit(train, cfg.log_vol_inputs, cfg.scale_target)
    xt, _, yt = sc.tensors(train, cfg.device)
    xv, _, yv = sc.tensors(val, cfg.device)
    model = PlainMLP(len(X_COLS), width, cfg.dropout).to(cfg.device)
    mse = torch.nn.functional.mse_loss
    epochs, _ = fit(model.parameters(), lambda: mse(model(xt), yt), lambda: mse(model(xv), yv).item(), model, cfg)
    model.eval()
    return model, sc, {"epochs": epochs}


@torch.no_grad()
def predict_moe(model: RGResMoE, sc: Scaler, df: pd.DataFrame, device: str, floor: float = 0.0):
    x, z, _ = sc.tensors(df, device)
    yhat, base, _, pi = model(x, z)
    clamp = lambda t: np.maximum(sc.unscale(t.cpu().numpy()), floor)
    return clamp(yhat), clamp(base), pi.cpu().numpy()


@torch.no_grad()
def predict_mlp(model: PlainMLP, sc: Scaler, df: pd.DataFrame, device: str, floor: float = 0.0) -> np.ndarray:
    x, _, _ = sc.tensors(df, device)
    return np.maximum(sc.unscale(model(x).cpu().numpy()), floor)


# ---------------------------------------------------------------- walk-forward
def windows(dates: pd.DatetimeIndex, cfg: TrainConfig) -> List[dict]:
    """Rolling development/test splits with a purge gap so targets never overlap the next split."""
    out, s = [], 0
    n_train = int(cfg.dev_days * (1 - cfg.val_frac))
    while s + cfg.dev_days < len(dates):
        dev = dates[s: s + cfg.dev_days]
        out.append({
            "train": dev[: n_train - cfg.purge_days],
            "val": dev[n_train: cfg.dev_days - cfg.purge_days],
            "test": dates[s + cfg.dev_days: s + cfg.dev_days + cfg.test_days],
        })
        s += cfg.step_days
    return out[-cfg.max_windows:] if cfg.max_windows else out


def rows(panel: pd.DataFrame, dates: pd.DatetimeIndex, need_y: bool = True) -> pd.DataFrame:
    df = panel.loc[panel.index.get_level_values("date").isin(dates)]
    return df.dropna(subset=[TARGET]) if need_y else df


def collapsed(m: dict) -> bool:
    """A run 'collapses' if its forecasts carry almost no ranking information."""
    return m["ic"] is None or m["ic"] < 0.1


def run(cfg: TrainConfig, save: bool = True, suffix: str = "", default_model: bool = False) -> dict:
    torch.set_num_threads(4)
    t0 = time.time()
    print("Building panel…")
    panel = build_panel(cfg)
    dates = panel.index.get_level_values("date").unique().sort_values()
    wins = windows(dates, cfg)
    print(f"  {len(panel):,} rows, {panel.index.get_level_values('symbol').nunique()} assets, "
          f"{len(dates)} dates, {len(wins)} walk-forward windows ({time.time() - t0:.1f}s)")

    moe_params = n_params(build_moe(len(X_COLS), len(Z_COLS), cfg))
    mlp_width = cfg.mlp_hidden or matched_width(len(X_COLS), moe_params)
    print(f"  RG-ResMoE: {moe_params:,} weights · size-matched MLP width {mlp_width} "
          f"({n_params(PlainMLP(len(X_COLS), mlp_width)):,} weights)")

    preds, window_log = [], []
    for wi, w in enumerate(wins):
        tw = time.time()
        train, val, test = rows(panel, w["train"]), rows(panel, w["val"]), rows(panel, w["test"])
        if test.empty:
            continue
        out = test[[TARGET, "is_crypto", "rv20", "mkt_vol", "idio_vol"]].copy()
        out["pred_naive"] = test["rv20"]  # baseline: last 20 days' realized vol
        info = {"window": wi, "test_start": str(w["test"][0].date()), "test_end": str(w["test"][-1].date()),
                "train_rows": len(train), "test_rows": len(test)}
        for seed in cfg.seeds:
            moe, sc, ep = train_rg_resmoe(train, val, cfg, seed)
            yhat, base, pi = predict_moe(moe, sc, test, cfg.device, cfg.vol_floor)
            out[f"pred_moe_s{seed}"] = yhat
            out[f"pred_base_s{seed}"] = base
            if seed == cfg.seeds[0]:
                for k in range(cfg.experts):
                    out[f"pi{k}"] = pi[:, k]
            mlp, sc2, ep2 = train_mlp(train, val, cfg, seed, mlp_width)
            out[f"pred_mlp_s{seed}"] = predict_mlp(mlp, sc2, test, cfg.device, cfg.vol_floor)
            info[f"epochs_s{seed}"] = {**ep, "mlp": ep2["epochs"]}
        preds.append(out)
        info["seconds"] = round(time.time() - tw, 1)
        m_moe = evaluate(out, f"pred_moe_s{cfg.seeds[0]}")
        m_mlp = evaluate(out, f"pred_mlp_s{cfg.seeds[0]}")
        m_nv = evaluate(out, "pred_naive")
        info.update({"ic_moe": m_moe["ic"], "ic_mlp": m_mlp["ic"], "ic_naive": m_nv["ic"]})
        window_log.append(info)
        print(f"  [{wi + 1:>2}/{len(wins)}] test {info['test_start']}…{info['test_end']}  "
              f"IC moe {m_moe['ic']:.3f} · mlp {m_mlp['ic']:.3f} · naive {m_nv['ic']:.3f}  "
              f"QLIKE moe {m_moe['qlike']:.3f} · mlp {m_mlp['qlike']:.3f} · naive {m_nv['qlike']:.3f}  ({info['seconds']}s)")

    oos = pd.concat(preds).sort_index()
    results = summarize(oos, cfg, window_log, moe_params, mlp_width)
    results["train_seconds"] = round(time.time() - t0, 1)

    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        tag = f"_{suffix}" if suffix else ""
        final = train_final(panel, dates, cfg, moe_params, MODEL_DIR / f"rg_resmoe{tag}.pt")
        if default_model:
            shutil.copy(MODEL_DIR / f"rg_resmoe{tag}.pt", MODEL_DIR / "rg_resmoe.pt")
        results["final_model"] = final
        (MODEL_DIR / f"walkforward_results{tag}.json").write_text(json.dumps(results, indent=2, default=str))
        oos.reset_index().to_csv(MODEL_DIR / f"walkforward_predictions{tag}.csv", index=False, float_format="%.5f")
        print(f"Saved models/rg_resmoe{tag}.pt, walkforward_results{tag}.json, walkforward_predictions{tag}.csv")
    print_summary(results)
    return results


def summarize(oos: pd.DataFrame, cfg: TrainConfig, window_log: List[dict], moe_params: int, mlp_width: int) -> dict:
    def per_model(col_prefix: str, mask=None) -> dict:
        runs = [evaluate(oos, f"{col_prefix}_s{s}", mask=mask) for s in cfg.seeds]
        agg = {}
        for key in ("ic", "rmse", "r2", "qlike"):
            vals = [r[key] for r in runs if r[key] is not None]
            agg[key] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)} if vals else None
        agg["collapsed_seeds"] = f"{sum(collapsed(r) for r in runs)}/{len(runs)}"
        return agg

    def block(mask=None) -> dict:
        return {
            "rg_resmoe": per_model("pred_moe", mask),
            "base_only": per_model("pred_base", mask),
            "plain_mlp": per_model("pred_mlp", mask),
            "naive_rv20": evaluate(oos, "pred_naive", mask=mask),
        }

    dates = oos.index.get_level_values("date")
    segments = {"all": block(), "stocks": block(~oos["is_crypto"]), "crypto": block(oos["is_crypto"])}
    for name, (a, b) in STRESS_PERIODS.items():
        mask = (dates >= a) & (dates <= b)
        if mask.sum() > 0:
            segments[name] = block(pd.Series(mask, index=oos.index))

    # Regime summary: when does each expert dominate? (first seed, stocks only)
    pi_cols = [f"pi{k}" for k in range(cfg.experts)]
    st = oos[~oos["is_crypto"]]
    dominant = st[pi_cols].to_numpy().argmax(1)
    experts = []
    for k in range(cfg.experts):
        sel = st[dominant == k]
        experts.append({
            "expert": k,
            "share_dominant": round(float((dominant == k).mean()), 4),
            "avg_weight": round(float(st[f"pi{k}"].mean()), 4),
            "mean_market_vol": round(float(sel["mkt_vol"].mean()), 4) if len(sel) else None,
            "mean_idio_vol": round(float(sel["idio_vol"].mean()), 4) if len(sel) else None,
            "mean_realized_vol": round(float(sel[TARGET].mean()), 4) if len(sel) else None,
        })
    timeline = st.groupby(level="date")[pi_cols + ["mkt_vol"]].mean().round(4)

    return {
        "paper": "Ye & Borde, Regime-Gated Residual Mixture-of-Experts, arXiv:2608.12251",
        "config": cfg.to_dict(),
        "weights": {"rg_resmoe": moe_params, "plain_mlp_width": mlp_width},
        "oos_period": [str(dates.min().date()), str(dates.max().date())],
        "oos_rows": int(len(oos)),
        "metrics": segments,
        "experts": experts,
        "regime_timeline": {"date": [str(d.date()) for d in timeline.index],
                            **{c: timeline[c].tolist() for c in pi_cols + ["mkt_vol"]}},
        "windows": window_log,
    }


def train_final(panel: pd.DataFrame, dates: pd.DatetimeIndex, cfg: TrainConfig, moe_params: int, path) -> dict:
    """Train on the most recent 504 days with known targets and save it for the app."""
    labelled = panel.dropna(subset=[TARGET]).index.get_level_values("date").unique().sort_values()
    dev = labelled[-cfg.dev_days:]
    n_train = int(cfg.dev_days * (1 - cfg.val_frac))
    train, val = rows(panel, dev[: n_train - cfg.purge_days]), rows(panel, dev[n_train:])
    model, sc, ep = train_rg_resmoe(train, val, cfg, cfg.seeds[0])
    meta = {
        "trained_from": str(dev[0].date()), "trained_through": str(dev[-1].date()),
        "x_cols": X_COLS, "z_cols": Z_COLS, "weights": moe_params, **ep,
    }
    torch.save({"state_dict": model.state_dict(), "scaler": sc.to_dict(), "config": cfg.to_dict(), "meta": meta},
               path)
    return meta


def print_summary(res: dict) -> None:
    print("\nOut-of-sample results", res["oos_period"], f"({res['oos_rows']:,} forecasts)")
    for seg, m in res["metrics"].items():
        print(f"\n  {seg}")
        print(f"    {'model':<12}{'IC':>16}{'RMSE':>16}{'QLIKE':>16}{'collapsed':>11}")
        for name in ("rg_resmoe", "plain_mlp", "base_only"):
            r = m[name]
            fmt = lambda k: f"{r[k]['mean']:.4f}±{r[k]['std']:.4f}" if r[k] else "–"
            print(f"    {name:<12}{fmt('ic'):>16}{fmt('rmse'):>16}{fmt('qlike'):>16}{r['collapsed_seeds']:>11}")
        n = m["naive_rv20"]
        print(f"    {'naive_rv20':<12}{n['ic']:>16.4f}{n['rmse']:>16.4f}{n['qlike']:>16.4f}")
    print("\n  Experts (stocks, seed 0): when each dominates")
    for e in res["experts"]:
        print(f"    expert {e['expert']}: dominant {e['share_dominant']:.0%} of rows · avg weight {e['avg_weight']:.2f} · "
              f"market vol {e['mean_market_vol']} · realized vol {e['mean_realized_vol']}")
    print(f"\n  Total time {res['train_seconds']}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RG-ResMoE with walk-forward evaluation.")
    ap.add_argument("--seeds", type=int, default=3, help="number of random seeds (default 3)")
    ap.add_argument("--windows", type=int, default=0, help="only the most recent N windows (0 = all)")
    ap.add_argument("--quick", action="store_true", help="smoke test: last 4 windows, 1 seed")
    ap.add_argument("--no-crypto", action="store_true", help="stocks only, like the paper")
    ap.add_argument("--preset", choices=["recommended", "paper", "ours"], default="recommended",
                    help="recommended = authors' settings + log inputs (default); paper = authors' exact settings; "
                         "ours = our original width-64 settings")
    ap.add_argument("--hidden", type=int, default=0, help="override expert width")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu or mps")
    ap.add_argument("--log-vol", dest="log_vol", action="store_true", default=None,
                    help="force log-scaled volatility inputs (on by default, except in --preset paper)")
    ap.add_argument("--raw-vol", dest="log_vol", action="store_false", help="use raw volatility inputs")
    ap.add_argument("--out", default="", help="output suffix (default: derived from the preset)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    common = dict(
        seeds=list(range(1 if a.quick else a.seeds)),
        max_windows=4 if a.quick else a.windows,
        include_crypto=not a.no_crypto, experts=a.experts, device=a.device,
    )
    if a.preset == "ours":
        cfg = TrainConfig(**common)
    else:
        cfg = paper_config(log_vol_inputs=(a.preset == "recommended"), **common)
    if a.hidden:
        cfg.hidden = a.hidden
    if a.log_vol is not None:
        cfg.log_vol_inputs = a.log_vol
    suffix = a.out or {"recommended": "paper_log", "paper": "paper", "ours": "ours"}[a.preset]
    run(cfg, save=not a.no_save, suffix=suffix, default_model=(a.preset == "recommended" and not a.out))


if __name__ == "__main__":
    main()
