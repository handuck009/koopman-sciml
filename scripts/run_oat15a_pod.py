"""Run full-snapshot POD analysis for the OAT15A transonic buffet dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oat15a_pod import PodConfig, run_full_pod_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "oat15a"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dt", type=float, default=0.03647971526)
    parser.add_argument("--snapshot-start-time", type=float, default=110.28)
    parser.add_argument("--modes-to-plot", type=int, default=8)
    parser.add_argument("--modes-to-save", type=int, default=20)
    parser.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures" / "oat15a_pod"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "oat15a_pod"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PodConfig(
        data_root=args.data_root,
        data_dir=args.data_dir,
        dt=args.dt,
        snapshot_start_time=args.snapshot_start_time,
        modes_to_plot=args.modes_to_plot,
        modes_to_save=args.modes_to_save,
    )
    summary = run_full_pod_analysis(
        config=config,
        figures_dir=Path(args.figures_dir),
        results_dir=Path(args.results_dir),
    )

    print("\nKey POD summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
