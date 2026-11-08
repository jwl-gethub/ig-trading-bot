"""Kronos forecasting worker: runs in its OWN process, started by bot.kronos_shadow.

Loads Kronos once, forecasts the next `pred_len` bars for every <MARKET>.csv in --inputs,
writes one JSON file, exits. Being a separate process means its memory (PyTorch, model)
is released afterwards and a crash or hang can never touch the trading loop.

    python -m bot.kronos_worker --inputs DIR --out FILE [--pred-len 6 --samples 5 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="NeoQuasar/Kronos-small")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--pred-len", type=int, default=6)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--bar-hours", type=float, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)

    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).parent / "vendor" / "kronos"))
    import pandas as pd
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    tokenizer = KronosTokenizer.from_pretrained(a.tokenizer)
    model = Kronos.from_pretrained(a.model)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    load_s = time.time() - t0

    results: dict[str, dict] = {}
    step = pd.Timedelta(hours=a.bar_hours)
    for csv in sorted(Path(a.inputs).glob("*.csv")):
        try:
            df = pd.read_csv(csv, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[["open", "high", "low", "close"]].astype(float).dropna().tail(a.lookback)
            x_ts = pd.Series(df.index)
            y_ts = pd.Series([df.index[-1] + step * (i + 1) for i in range(a.pred_len)])
            torch.manual_seed(a.seed)
            pred = predictor.predict(df.reset_index(drop=True), x_ts, y_ts, a.pred_len,
                                     T=1.0, top_p=0.9, sample_count=a.samples, verbose=False)
            last = float(df["close"].iloc[-1])
            closes = [round(float(c), 5) for c in pred["close"]]
            results[csv.stem] = {
                "bar": str(df.index[-1]), "last_close": last, "pred_close": closes[-1], "pred_path": closes,
                "pred_high": round(float(pred["high"].max()), 5), "pred_low": round(float(pred["low"].min()), 5),
                "exp_ret_pct": round(100 * (closes[-1] / last - 1), 4),
            }
        except Exception as exc:  # one bad market must not stop the others
            results[csv.stem] = {"error": repr(exc)}
    Path(a.out).write_text(json.dumps({"load_seconds": round(load_s, 1),
                                       "seconds": round(time.time() - t0, 1), "results": results}))


if __name__ == "__main__":
    main()
