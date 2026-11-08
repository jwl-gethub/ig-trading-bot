"""Stand-in for bot.kronos_worker in tests: same CLI, no PyTorch. Predicts +0.5% for every market."""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
for a in ("--inputs", "--out", "--model", "--tokenizer", "--pred-len", "--samples", "--lookback", "--bar-hours"):
    ap.add_argument(a)
ap.add_argument("--mode", default="ok")
a, _ = ap.parse_known_args()
mode = Path(a.inputs).parent.joinpath("mode").read_text() if Path(a.inputs).parent.joinpath("mode").exists() else "ok"
if "FAKE_KRONOS_MODE" in __import__("os").environ:
    mode = __import__("os").environ["FAKE_KRONOS_MODE"]
if mode == "crash":
    sys.exit(3)
if mode == "hang":
    time.sleep(30)
res = {}
for csv in sorted(Path(a.inputs).glob("*.csv")):
    df = pd.read_csv(csv, index_col=0)
    last = float(df["close"].iloc[-1])
    res[csv.stem] = {"bar": str(pd.to_datetime(df.index[-1], utc=True)), "last_close": last,
                     "pred_close": last * 1.005, "pred_path": [last * 1.005], "pred_high": last * 1.01,
                     "pred_low": last * 0.99, "exp_ret_pct": 0.5}
Path(a.out).write_text(json.dumps({"load_seconds": 0.1, "seconds": 0.2, "results": res}))
