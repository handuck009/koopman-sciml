"""Run stride-sampled POD-DMD analysis for the OAT15A buffet dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oat15a_pod_dmd import ComparisonConfig, run_stride_pod_dmd_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "oat15a"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--base-dt", type=float, default=0.03647971526)
    parser.add_argument("--snapshot-start-time", type=float, default=110.28)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--pod-rank", type=int, default=5)
    parser.add_argument("--dmd-fit-rank", type=int, default=40)
    parser.add_argument("--dmd-pair-count", type=int, default=4)
    parser.add_argument(
        "--figures-dir",
        default=str(PROJECT_ROOT / "figures" / "oat15a_pod_dmd_stride2"),
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results" / "oat15a_pod_dmd_stride2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ComparisonConfig(
        data_root=args.data_root,
        data_dir=args.data_dir,
        base_dt=args.base_dt,
        snapshot_start_time=args.snapshot_start_time,
        stride=args.stride,
        pod_rank=args.pod_rank,
        dmd_fit_rank=args.dmd_fit_rank,
        dmd_pair_count=args.dmd_pair_count,
    )
    summary = run_stride_pod_dmd_analysis(
        config=config,
        figures_dir=Path(args.figures_dir),
        results_dir=Path(args.results_dir),
    )
    print("\nKey POD-DMD summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
