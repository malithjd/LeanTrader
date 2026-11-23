"""Does news help? Change in Rank IC of each news variant vs the baseline, same ranker.

Owner: MJ. Spec: "Evals -> Ranking": change vs factors-only with a 95% block-bootstrap
interval (20-day blocks).

Reads data/results/preds_{stage}_{model}_{variant}.csv and writes
data/results/compare_{stage}.csv: one row per (model, variant) vs (model, base).

Usage:
    python compare.py            # validation
    python compare.py --test     # test run
"""

from __future__ import annotations

import argparse

import pandas as pd

from metrics import daily_ic, ic_diff_ci
from walkforward import RESULTS


def compare(stage: str) -> pd.DataFrame:
    rows = []
    for model in ["lgbm", "gru"]:
        base_path = RESULTS / f"preds_{stage}_{model}_base.csv"
        if not base_path.exists():
            continue
        base_ic = daily_ic(pd.read_csv(base_path))
        for path in sorted(RESULTS.glob(f"preds_{stage}_{model}_base+*.csv")):
            variant = path.stem.removeprefix(f"preds_{stage}_{model}_")
            if variant.endswith("_nocovid"):
                continue
            d = ic_diff_ci(daily_ic(pd.read_csv(path)), base_ic)
            rows.append({"model": model, "variant": variant, "vs": "base", **d})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"compare_{stage}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    out = compare("test" if args.test else "val")
    print(out.round(4).to_string(index=False) if not out.empty else "No news variants to compare yet.")


if __name__ == "__main__":
    main()
