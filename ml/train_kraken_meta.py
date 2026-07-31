"""Train and evaluate the LightGBM meta-labeler with walk-forward validation."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from strategies.ml_meta import (
    MLMetaResult,
    walk_forward_validation,
    train_and_export,
    predict_meta_label,
    build_hypothesis_with_meta_label,
)
from strategies.kraken_swing import generate_signal
from strategies.pipeline import Hypothesis

OUTDIR = os.environ.get("AUDIT_DIR", "/Users/vera/trading-system/backtests")
STATE_DIR = os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")


def _snapshot_from_df(df: pd.DataFrame, idx: int):
    closes = tuple(float(x) for x in df["close"].iloc[max(0, idx - 49) : idx + 1])
    highs = tuple(float(x) for x in df["high"].iloc[max(0, idx - 49) : idx + 1])
    lows = tuple(float(x) for x in df["low"].iloc[max(0, idx - 49) : idx + 1])
    volumes = tuple(float(x) for x in df["volume"].iloc[max(0, idx - 49) : idx + 1])
    tp = (np.array(highs) + np.array(lows) + np.array(closes)) / 3.0
    vwap = float(np.cumsum(tp * volumes)[-1] / max(np.cumsum(volumes)[-1], 1e-9))
    ts = str(df["timestamp"].iloc[idx])
    return type("Snap", (), {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "vwap": float(vwap),
        "timestamp": ts,
    })()


def main() -> int:
    from ml.data import load_or_hydrate

    df = load_or_hydrate()
    if df is None or len(df) < 120:
        print("insufficient_history")
        return 1

    # preserve timestamp for backtest compatibility
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    wf = walk_forward_validation(df, folds=6, holding_bars=3)
    print("walk_forward=" + json.dumps(wf, indent=2))

    if wf.get("status") == "skipped":
        print("walk_forward_skipped")
        return 0

    model, manifest = train_and_export(df, folds=6, holding_bars=3)
    print("model_manifest=" + json.dumps(manifest, indent=2))

    tail = df.tail(30).reset_index(drop=True)
    if len(tail) == 0:
        print("no_tail_data")
        return 0

    rule_signals = 0
    ml_blocks = 0
    ml_adopts = 0
    from strategies.ml_meta import _no_lookahead_features

    for i in range(len(tail)):
        snap_idx = int(tail.index[i])
        snap = _snapshot_from_df(df, snap_idx)
        sig = generate_signal(snap)
        if sig is None:
            continue
        rule_hyp = Hypothesis(
            direction=sig.direction,
            score=sig.score,
            evidence=tuple(sig.meta.get("regime", "RULE").split()),
            strategy="kraken-swing",
        )
        window_to_idx = df.loc[:snap_idx]
        feats = _no_lookahead_features(window_to_idx)
        if feats.empty:
            continue
        label_prob = predict_meta_label(model, feats)[1]
        ml_result = MLMetaResult(
            adopted=True,
            model_version=manifest["version"],
            confidence=label_prob,
            precision=manifest["metrics"]["precision"],
            recall=manifest["metrics"]["recall"],
            f1=manifest["metrics"]["f1"],
            auc=manifest["metrics"]["auc"],
            walk_forward_summary={"mean_f1": wf.get("mean_f1")},
            evidence=(),
        )
        final_hyp = build_hypothesis_with_meta_label(rule_hyp, model, feats, ml_result)
        if rule_hyp.direction != "hold":
            rule_signals += 1
            if final_hyp.direction == "hold":
                ml_blocks += 1
            else:
                ml_adopts += 1

    print(f"rule_signals={rule_signals}")
    print(f"ml_adopts={ml_adopts}")
    print(f"ml_blocks={ml_blocks}")
    out = {
        "module": "kraken-ml-meta",
        "symbol": "LINKUSD",
        "walk_forward": wf,
        "model": manifest,
        "recent_30_bars": {
            "rule_signals": rule_signals,
            "ml_adopts": ml_adopts,
            "ml_blocks": ml_blocks,
        },
    }
    path = os.path.join(OUTDIR, "kraken_ml_meta.json")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved=" + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
