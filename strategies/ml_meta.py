"""LightGBM meta-labeling for the hypothesis layer."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from strategies.pipeline import Hypothesis

SYMBOL = "LINKUSD"
INTERVAL = "1h"
STATE_DIR = Path(os.environ.get("STATE_DIR", Path(__file__).resolve().parent.parent / "state"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", STATE_DIR / "models"))
# Persistence note: Railway has a detached `kraken-agent-volume` mounted at `/app/state`.
# To preserve trained models across deploys, set `MODEL_DIR=/app/state/models` in Railway env.
# the volume there in Railway service config.


@dataclass(frozen=True)
class MLMetaResult:
    adopted: bool
    model_version: str
    confidence: float
    precision: float | None
    recall: float | None
    f1: float | None
    auc: float | None
    walk_forward_summary: dict[str, Any]
    evidence: tuple[str, ...] = ()
    adopted_by_council: bool = False
    audit_ref: str | None = None


def _ensure_dirs() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def _default_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )


def _no_lookahead_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"close": df["close"].values}, index=df.index)
    out["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    out["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    out["vwap"] = pv.cumsum() / df["volume"].cumsum().replace(0, 1e-12)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = tr1.iloc[0]
    out["atr"] = tr.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    out["rsi"] = 100.0 - 100.0 / (1.0 + rs)
    out["rsi"] = out["rsi"].fillna(50.0)

    out["ret_1"] = df["close"].pct_change(1)
    out["ret_3"] = df["close"].pct_change(3)
    out["ret_6"] = df["close"].pct_change(6)
    out["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, 1e-12)

    out["price_vs_vwap"] = (df["close"] - out["vwap"]) / df["close"]
    return out.dropna()


def _make_label(feature_df: pd.DataFrame, holding_bars: int = 3, threshold: float = 0.0006) -> pd.Series:
    fwd = feature_df["close"].shift(-holding_bars) / feature_df["close"] - 1.0
    labels = (fwd > threshold).astype(float)
    labels[fwd.isna()] = np.nan
    return labels


def _dataset(frop, holding_bars: int = 3):
    features = _no_lookahead_features(frop)
    labels = _make_label(features, holding_bars=holding_bars)
    aligned = features.join(labels.rename("label")).dropna()
    return aligned


def _summarize_folds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    auc_values = [r["auc"] for r in rows if r.get("auc") is not None]
    return {
        "status": "ok",
        "folds": len(rows),
        "fold_summaries": rows[-3:],
        "mean_precision": float(np.mean([r["precision"] for r in rows])) if rows else None,
        "mean_recall": float(np.mean([r["recall"] for r in rows])) if rows else None,
        "mean_f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
        "mean_auc": float(np.mean(auc_values)) if auc_values else None,
    }


def walk_forward_validation(df: pd.DataFrame, folds: int = 6, holding_bars: int = 3) -> dict[str, Any]:
    aligned = _dataset(df, holding_bars=holding_bars)
    if len(aligned) < folds * 60:
        return {
            "status": "skipped",
            "reason": "insufficient_history",
            "bars": int(len(aligned)),
            "required_bars": int(folds * 60),
        }

    x = aligned.drop(columns=["label", "close"], errors="ignore").values
    y = aligned["label"].values
    fold_size = len(aligned) // folds
    rows: list[dict[str, Any]] = []
    best_model = _default_model()

    for i in range(1, folds):
        train_end = i * fold_size
        if train_end < 60:
            continue
        x_train, y_train = x[:train_end], y[:train_end]
        x_test, y_test = x[train_end:train_end + fold_size], y[train_end:train_end + fold_size]
        if len(x_test) == 0:
            continue

        model = _default_model()
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append(
            {
                "fold": i,
                "train_bars": int(len(x_train)),
                "test_bars": int(len(x_test)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) == 2 else None,
                "positive_rate": float(y_test.mean()),
                "predicted_positive_rate": float(pred.mean()),
            }
        )
        best_model = model

    return _summarize_folds(rows)


def train_and_export(df: pd.DataFrame, folds: int = 6, holding_bars: int = 3) -> tuple[LGBMClassifier, dict[str, Any]]:
    aligned = _dataset(df, holding_bars=holding_bars)
    if len(aligned) < 200:
        raise ValueError("insufficient_history_for_training")

    x = aligned.drop(columns=["label", "close"], errors="ignore").values
    y = aligned["label"].values
    split = int(len(aligned) * 0.8)
    model = _default_model()
    model.fit(x[:split], y[:split])
    val_pred = model.predict_proba(x[split:])[:, 1]
    val_label = y[split:]
    metrics = {
        "precision": float(precision_score(val_label, (val_pred >= 0.5).astype(int), zero_division=0)),
        "recall": float(recall_score(val_label, (val_pred >= 0.5).astype(int), zero_division=0)),
        "f1": float(f1_score(val_label, (val_pred >= 0.5).astype(int), zero_division=0)),
        "auc": float(roc_auc_score(val_label, val_pred)) if len(np.unique(val_label)) == 2 else None,
    }
    _ensure_dirs()
    version = f"lgbm_meta_v{int(pd.Timestamp.utcnow().timestamp())}"
    path = MODEL_DIR / f"{version}.txt"
    model.booster_.save_model(path)
    manifest = {
        "version": version,
        "path": str(path),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "holding_bars": holding_bars,
        "train_bars": int(split),
        "validation_bars": int(len(aligned) - split),
        "metrics": metrics,
        "feature_names": [c for c in aligned.drop(columns=[ "label", "close"], errors="ignore").columns],
    }
    with open(MODEL_DIR / f"{version}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return model, manifest


def load_latest_model() -> tuple[Any, dict[str, Any] | None]:
    if not os.path.isdir(MODEL_DIR):
        return None, None
    candidates = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("lgbm_meta_") and f.endswith(".txt")])
    if not candidates:
        return None, None
    latest = candidates[-1]
    booster = Booster(model_file=MODEL_DIR / latest)
    version = latest.replace(".txt", "")
    manifest_path = MODEL_DIR / f"{version}.json"
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    return booster, manifest


def predict_meta_label(booster: Any, current_features: pd.DataFrame) -> tuple[int, float]:
    row = current_features.drop(columns=["label", "close"], errors="ignore").values[-1:].astype(float)
    prob = float(booster.predict(row)[0])
    label = int(prob >= 0.5)
    return label, prob


def build_hypothesis_with_meta_label(
    rule_hypothesis: Hypothesis,
    booster: Any,
    current_features: pd.DataFrame,
    meta_result: MLMetaResult,
) -> Hypothesis:
    label, confidence = predict_meta_label(booster, current_features)
    adopted = bool(label == 1 and meta_result.adopted)
    if not adopted:
        return Hypothesis(
            direction="hold",
            score=0,
            evidence=("meta-label blocked",) + meta_result.evidence,
            strategy="ml-meta-blocked",
        )

    base_score = rule_hypothesis.score
    ml_boost = int(confidence * 25)
    score = min(base_score + ml_boost, 100)
    return Hypothesis(
        direction=rule_hypothesis.direction,
        score=score,
        evidence=rule_hypothesis.evidence
        + (
            f"ml-meta-label={label}",
            f"ml-confidence={confidence:.3f}",
            f"ml-version={meta_result.model_version}",
        ),
        strategy="ml-meta-enhanced",
    )


__all__ = [
    "MLMetaResult",
    "walk_forward_validation",
    "train_and_export",
    "load_latest_model",
    "predict_meta_label",
    "build_hypothesis_with_meta_label",
]
