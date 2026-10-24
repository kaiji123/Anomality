"""Feature engineering, anomaly detection and the decision model.

Pipeline for one symbol:
  1. features()   - backward-looking indicators for every day (no look-ahead).
  2. detect()     - flags historical anomalies using robust z-scores plus an
                    Isolation Forest over several features at once.
  3. DecisionModel - a decision tree that learns which market conditions tend to
                    come right before an unusual move (a "shock" day) within the
                    next `horizon` days, and scores today's conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

FEATURES: Dict[str, str] = {
    "ret_z": "Return z-score (today)",
    "abs_ret_z": "Move size |z| (today)",
    "vol_z": "Volume z-score",
    "range_z": "Intraday range z-score",
    "gap_z": "Opening gap z-score",
    "vol_regime": "Volatility regime (5d / 60d)",
    "mom_5": "5-day momentum (z)",
    "trend_50": "Distance from 50-day avg",
    "rsi_14": "RSI (14)",
    "bb_pctb": "Bollinger %B",
    "shocks_10": "Shocks in last 10 days",
    "vol_trend": "Volume trend (5d / 60d)",
}


@dataclass
class Settings:
    threshold: float = 2.5  # |z| above which a day counts as a shock / anomaly
    horizon: int = 5  # look-ahead window (days) for the decision model
    depth: int = 4  # max depth of the decision tree
    contamination: float = 0.02  # share of days the Isolation Forest flags
    model: str = "forest"  # "forest" (ensemble of trees, scores) or "tree" (single tree)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """z-score of today's value vs. the previous `window` days (today excluded)."""
    mean = s.rolling(window).mean().shift(1)
    std = s.rolling(window).std().shift(1)
    return (s - mean) / std.replace(0, np.nan)


def features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    ret = np.log(close).diff()
    vol20 = ret.rolling(20).std().shift(1)
    out["ret"] = close.pct_change()
    out["log_ret"] = ret
    out["vol20"] = vol20

    out["ret_z"] = ret / vol20
    out["abs_ret_z"] = out["ret_z"].abs()
    log_volume = np.log1p(out["volume"])
    out["vol_z"] = _zscore(log_volume, 20)
    out["range_z"] = _zscore((out["high"] - out["low"]) / close, 20)
    out["gap_z"] = np.log(out["open"] / close.shift(1)) / vol20
    out["vol_regime"] = ret.rolling(5).std() / ret.rolling(60).std()
    out["mom_5"] = np.log(close / close.shift(5)) / (vol20 * np.sqrt(5))
    out["trend_50"] = close / close.rolling(50).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out["bb_pctb"] = (close - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)

    shock = (out["abs_ret_z"] > threshold).astype(float)
    out["shock"] = shock.where(out["ret_z"].notna())
    out["shocks_10"] = shock.rolling(10).sum()
    vol_mean = out["volume"].replace(0, np.nan)
    out["vol_trend"] = vol_mean.rolling(5).mean() / vol_mean.rolling(60).mean()

    for col in ("vol_z", "vol_trend"):
        # Some indices report no volume; treat that as "nothing unusual".
        if out["volume"].sum() == 0:
            out[col] = 0.0 if col == "vol_z" else 1.0
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.iloc[60:]  # drop warm-up rows where long windows are empty


def detect(fx: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Add anomaly flags, a 0-100 anomaly score and a human-readable type."""
    cols = ["ret_z", "vol_z", "range_z", "gap_z", "vol_regime"]
    X = fx[cols].fillna(0).clip(-15, 15).to_numpy()
    iso = IsolationForest(n_estimators=200, contamination=s.contamination, random_state=7)
    iso.fit(X)
    raw = -iso.score_samples(X)  # higher = more isolated
    lo, hi = np.percentile(raw, 50), raw.max()
    iso_strength = np.clip((raw - lo) / max(hi - lo, 1e-9), 0, 1)
    iso_flag = iso.predict(X) == -1

    thr = s.threshold
    z_max = np.nanmax(
        np.vstack([fx["abs_ret_z"], fx["vol_z"].clip(lower=0), fx["range_z"].clip(lower=0), fx["gap_z"].abs()]),
        axis=0,
    )
    z_max = np.nan_to_num(z_max)
    stat_score = np.clip(50 * z_max / thr, 0, 100)
    iso_score = np.where(iso_flag, 50 + 50 * iso_strength, 50 * iso_strength)
    fx = fx.copy()
    fx["iso_flag"] = iso_flag
    fx["anomaly_score"] = np.maximum(stat_score, iso_score).round(1)

    types: List[Optional[str]] = []
    for r_z, v_z, g_z, rg_z, iso_f in zip(fx["ret_z"], fx["vol_z"], fx["gap_z"], fx["range_z"], iso_flag):
        tags = []
        if r_z is not None and abs(r_z) > thr:
            tags.append("Price spike" if r_z > 0 else "Price crash")
        if v_z is not None and v_z > thr:
            tags.append("Volume surge")
        if g_z is not None and abs(g_z) > thr:
            tags.append("Gap " + ("up" if g_z > 0 else "down"))
        if rg_z is not None and rg_z > thr and not tags:
            tags.append("Wide range")
        if iso_f and not tags:
            tags.append("Multi-factor outlier")
        types.append(" + ".join(tags) if tags else None)
    fx["anomaly_type"] = types
    fx["is_anomaly"] = fx["anomaly_type"].notna()
    return fx


class DecisionModel:
    """Predicts an unusual move within the next `horizon` days.

    The risk score comes from a random forest (an ensemble of decision trees,
    more robust) or a single decision tree, depending on settings. A single
    shallow tree is always fitted too, so every verdict comes with readable rules.
    """

    def __init__(self, s: Settings):
        self.s = s
        self.cols = list(FEATURES)

    def _target(self, fx: pd.DataFrame) -> pd.Series:
        h = self.s.horizon
        shock = fx["shock"].fillna(0)
        future = pd.concat([shock.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1)
        future.iloc[-h:] = np.nan  # future not known yet
        return future

    def _tree(self, n: int) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.s.depth,
            min_samples_leaf=max(15, int(n * 0.02)),
            class_weight="balanced",
            random_state=0,
        )

    def _scorer(self, n: int):
        if self.s.model == "tree":
            return self._tree(n)
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=self.s.depth + 1,
            min_samples_leaf=max(10, int(n * 0.02)),
            class_weight="balanced",
            n_jobs=2,
            random_state=0,
        )

    def run(self, fx: pd.DataFrame) -> dict:
        X_all = fx[self.cols].fillna(fx[self.cols].median()).fillna(0)
        y_all = self._target(fx)
        known = y_all.notna()
        X, y = X_all[known], y_all[known].astype(int)
        base_rate = float(y.mean())
        if y.nunique() < 2 or y.sum() < 10:
            raise ValueError("Too few unusual moves in history to train the model. Lower the sensitivity threshold.")

        # 1) Honest evaluation: train on the first 75% of history, test on the rest.
        split = int(len(X) * 0.75)
        val_model = self._scorer(split).fit(X.iloc[:split], y.iloc[:split])
        y_test = y.iloc[split:]
        p_test = val_model.predict_proba(X.iloc[split:])[:, 1]
        pred_test = (p_test >= 0.5).astype(int)
        metrics = {
            "train_days": split,
            "test_days": int(len(y_test)),
            "test_start": X.index[split].strftime("%Y-%m-%d"),
            "base_rate": round(base_rate, 4),
            "test_base_rate": round(float(y_test.mean()), 4),
            "accuracy": round(float(accuracy_score(y_test, pred_test)), 3),
            "precision": round(float(precision_score(y_test, pred_test, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, pred_test, zero_division=0)), 3),
            "auc": round(float(roc_auc_score(y_test, p_test)), 3) if y_test.nunique() > 1 else None,
        }

        # 2) Final model: train on all labelled history, then score today.
        tree = self._tree(len(X)).fit(X, y)
        scorer = tree if self.s.model == "tree" else self._scorer(len(X)).fit(X, y)
        p_all = scorer.predict_proba(X_all)[:, 1]
        latest = X_all.iloc[[-1]]
        p_now = float(p_all[-1])

        # Unweighted stats per node (how often similar days were followed by a shock).
        ind = tree.decision_path(X)
        node_n = np.asarray(ind.sum(axis=0)).ravel()
        node_pos = np.asarray(ind.T @ y.to_numpy()).ravel()
        leaf = int(tree.apply(latest)[0])
        leaf_n, leaf_pos = int(node_n[leaf]), int(node_pos[leaf])
        leaf_rate = leaf_pos / leaf_n if leaf_n else 0.0

        score = round(100 * p_now, 1)
        verdict = "HIGH" if score >= 65 else "ELEVATED" if score >= 50 else "NORMAL"

        risk_history = [
            {"date": d.strftime("%Y-%m-%d"), "risk": round(100 * p, 1), "oos": True}
            for d, p in zip(X.index[split:], p_test)
        ]
        for d in X_all.index[~known.to_numpy()]:
            risk_history.append({"date": d.strftime("%Y-%m-%d"), "risk": round(100 * p_all[X_all.index.get_loc(d)], 1), "oos": False})

        return {
            "risk_score": score,
            "verdict": verdict,
            "horizon": self.s.horizon,
            "threshold": self.s.threshold,
            "leaf": {"days": leaf_n, "followed_by_shock": leaf_pos, "rate": round(leaf_rate, 4),
                     "lift": round(leaf_rate / base_rate, 2) if base_rate else None},
            "path": self._path(tree, latest, node_n, node_pos),
            "tree": self._export(tree, node_n, node_pos, set(tree.decision_path(latest).indices.tolist())),
            "model": self.s.model,
            "importance": sorted(
                [{"feature": c, "label": FEATURES[c], "value": round(float(v), 4)}
                 for c, v in zip(self.cols, scorer.feature_importances_) if v > 0.005],
                key=lambda r: -r["value"],
            ),
            "current_features": {c: (None if pd.isna(v) else round(float(v), 3)) for c, v in fx[self.cols].iloc[-1].items()},
            "metrics": metrics,
            "risk_history": risk_history,
        }

    def _path(self, tree, latest, node_n, node_pos) -> List[dict]:
        t = tree.tree_
        steps = []
        for node in tree.decision_path(latest).indices:
            f = t.feature[node]
            if f < 0:
                continue
            col = self.cols[f]
            value = float(latest.iloc[0, f])
            thr = float(t.threshold[node])
            steps.append({
                "feature": col, "label": FEATURES[col], "value": round(value, 3),
                "threshold": round(thr, 3), "op": "<=" if value <= thr else ">",
                "node_rate": round(node_pos[node] / node_n[node], 4) if node_n[node] else None,
            })
        return steps

    def _export(self, tree, node_n, node_pos, active: set) -> dict:
        t = tree.tree_

        def build(node: int) -> dict:
            n = int(node_n[node])
            item = {
                "id": node, "days": n, "rate": round(node_pos[node] / n, 4) if n else 0,
                "active": node in active,
            }
            if t.children_left[node] == -1:
                item["leaf"] = True
                return item
            col = self.cols[t.feature[node]]
            item.update({
                "feature": col, "label": FEATURES[col], "threshold": round(float(t.threshold[node]), 3),
                "left": build(int(t.children_left[node])), "right": build(int(t.children_right[node])),
            })
            return item

        return build(0)
