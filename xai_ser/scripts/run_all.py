#!/usr/bin/env python
"""Chain the analysis stages end to end.

Assumes the audio already exists (`scripts/fetch_cremad.sh` then
`python -m xai_ser.transforms`). Each stage writes its own timestamped output
directory, so re-running never overwrites a previous run's results.
"""

from __future__ import annotations

import argparse
import time

from xai_ser import ann, eda, extract, models, xai_deep, xai_tabular


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-conditions", nargs="*", default=["ref"])
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    stages = []
    if not args.skip_extract:
        stages.append(("extract", lambda: extract.run(workers=args.workers)))
    stages += [
        ("eda", eda.run),
        ("models", lambda: models.run(train_conditions=tuple(args.train_conditions), tune=args.tune)),
        ("ann", ann.train_ann),
        ("xai_tabular", xai_tabular.run),
        ("xai_deep", xai_deep.run),
    ]

    for name, fn in stages:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        t0 = time.time()
        fn()
        print(f"[{name}] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
