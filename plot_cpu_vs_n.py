#!/usr/bin/env python3
"""
Plot CPU time vs n from ./sptrsv -p output (CSV: n,cpu_time_ms).

Usage:
  python3 plot_cpu_vs_n.py cpu_vs_n.csv
  python3 plot_cpu_vs_n.py cpu_vs_n.csv -o myplot.png

Requires: pip install matplotlib
"""

from __future__ import annotations

import argparse
import csv
import sys


def load_csv(path: str) -> tuple[list[int], list[float]]:
    ns: list[int] = []
    ts: list[float] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError("CSV must have header: n,cpu_time_ms")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            ns.append(int(row[0]))
            ts.append(float(row[1]))
    if not ns:
        raise ValueError("no data rows")
    return ns, ts


def main() -> int:
    p = argparse.ArgumentParser(description="Plot sptrsv_cpu time vs n from profile CSV")
    p.add_argument(
        "csv",
        nargs="?",
        default="cpu_vs_n.csv",
        help="path to CSV (default: cpu_vs_n.csv)",
    )
    p.add_argument("-o", "--output", default="cpu_vs_n.png", help="output image path")
    p.add_argument("--show", action="store_true", help="open interactive window (if display available)")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Missing matplotlib. Install with:  pip install matplotlib", file=sys.stderr)
        return 1

    try:
        ns, ts = load_csv(args.csv)
    except OSError as e:
        print(f"Cannot read {args.csv!r}: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Invalid CSV: {e}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, ts, "o-", color="#0d47a1", linewidth=2, markersize=7, markerfacecolor="#1976d2")
    ax.set_xlabel("n (size of L: n×n)")
    ax.set_ylabel("CPU time (ms)")
    ax.set_title("sptrsv_cpu runtime vs n  (profile: dense lower-tri L, nB = 128)")
    ax.grid(True, alpha=0.35)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
