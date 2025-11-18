from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

from momfpriors.benchmarks import BENCHMARKS

if TYPE_CHECKING:
    from matplotlib.path import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main(
    list: str,
    data_dir: str | Path | None = None,
):
    """List available benchmarks."""
    match list:
        case "all":
            logger.info(BENCHMARKS(datadir=data_dir).keys())
        case _:
            logger.error(f"Invalid list type: {list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        required=True,
        choices=["all"],
        help="List all Benchmarks or filter by type"
        "\nall: list all Benchmarks"
    )
    parser.add_argument(
        "--data_dir", "-data",
        type=str,
        default=None,
        help="Path to the benchmark data directory."
    )
    args = parser.parse_args()
    main(
        list=args.list,
        data_dir=args.data_dir,
    )
