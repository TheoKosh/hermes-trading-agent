"""Train separate paper-only LightGBM candidates for futures contracts."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SYMBOLS = ("MNQ=F", "MYM=F", "MES=F")
INTERVAL = "15m"
DAYS = 45


def _paths() -> tuple[Path, Path]:
    state = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    models = Path(os.environ.get("FUTURES_MODEL_DIR", str(state / "models" / "futures")))
    audit = Path(os.environ.get("AUDIT_DIR", str(state / "backtests")))
    models.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    return models, audit


def _frame(symbol: str) -> pd.DataFrame:
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    raw = yf.download(symbol, start=start, end=datetime.now(timezone.utc), interval=INTERVAL, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(symbol, axis=1, level=1)
    df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})[["open", "high", "low", "close", "volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df[(df["close"] > 0) & (df["volume"] >= 0)].copy()


def _dataset(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    x["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    delta = df["close"].diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = gain.ewm(alpha=1 / 14, adjust=False).mean() / loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, 1e-12)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["ret_1"] = df["close"].pct_change(1)
    x["ret_3"] = df["close"].pct_change(3)
    x["ret_6"] = df["close"].pct_change(6)
    x["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, 1e-12)
    fwd = df["close"].shift(-3) / df["close"] - 1
    x["label"] = (fwd > 0.0006).astype(float)
    x.loc[fwd.isna(), "label"] = np.nan
    return x.dropna()


def _train_symbol(symbol: str) -> dict[str, Any]:
    df = _frame(symbol)
    data = _dataset(df)
    if len(data) < 240:
        return {"symbol": symbol, "status": "skipped", "reason": "insufficient_history", "bars": int(len(data))}
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    features = [c for c in data.columns if c != "label"]
    model = LGBMClassifier(n_estimators=120, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=4, verbose=-1)
    model.fit(train[features], train["label"])
    prob = model.predict_proba(test[features])[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {"precision": float(precision_score(test["label"], pred, zero_division=0)), "recall": float(recall_score(test["label"], pred, zero_division=0)), "f1": float(f1_score(test["label"], pred, zero_division=0)), "auc": float(roc_auc_score(test["label"], prob)) if len(np.unique(test["label"])) == 2 else None}
    models, _ = _paths()
    safe = symbol.replace("=", "_")
    version = f"futures_meta_{safe}_{int(time.time())}"
    model_path = models / f"{version}.txt"
    manifest_path = models / f"{version}.json"
    model.booster_.save_model(str(model_path))
    manifest = {"version": version, "symbol": symbol, "interval": INTERVAL, "asset_class": "futures", "train_bars": int(len(train)), "validation_bars": int(len(test)), "metrics": metrics, "path": str(model_path), "feature_names": features}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"symbol": symbol, "status": "complete", "manifest": manifest}


def main() -> int:
    _, audit = _paths()
    results = [_train_symbol(symbol) for symbol in SYMBOLS]
    (audit / "futures_ml_meta.json").write_text(json.dumps({"module": "futures-ml-meta", "results": results}, indent=2))
    print(json.dumps({"module": "futures-ml-meta", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
