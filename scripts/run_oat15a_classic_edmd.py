"""Run classical polynomial EDMD on POD-reduced OAT15A buffet data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oat15a_classic_edmd import ClassicEDMDConfig, run_classic_edmd_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "oat15a"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--pod-dmd-results",
        default=str(PROJECT_ROOT / "results" / "oat15a_pod_dmd_stride2" / "oat15a_stride2_pod_dmd_results.npz"),
    )
    parser.add_argument("--selected-degree", type=int, default=3)
    parser.add_argument(
        "--figures-dir",
        default=str(PROJECT_ROOT / "figures" / "oat15a_classic_edmd"),
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results" / "oat15a_classic_edmd"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ClassicEDMDConfig(
        data_root=args.data_root,
        data_dir=args.data_dir,
        pod_dmd_results=args.pod_dmd_results,
        selected_degree=args.selected_degree,
    )
    summary = run_classic_edmd_analysis(
        config,
        figures_dir=Path(args.figures_dir),
        results_dir=Path(args.results_dir),
    )
    print("\nKey classical EDMD summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
