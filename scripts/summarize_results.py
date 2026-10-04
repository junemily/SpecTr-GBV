import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_metrics(path: Path):
    metrics = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("FAIR_METRICS "):
                metrics.append(json.loads(line[len("FAIR_METRICS ") :]))
    return metrics


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Summarize FAIR_METRICS logs.")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in args.logs:
        for metric in read_metrics(path):
            groups[metric["method"]].append(metric)

    print("| Method | Runs | Tokens/s | Block efficiency | Acceptance rate |")
    print("|---|---:|---:|---:|---:|")
    for method in sorted(groups):
        rows = groups[method]
        speed_mean, speed_std = mean_std([row["tokens_per_second"] for row in rows])
        block_mean, block_std = mean_std([row["block_efficiency"] for row in rows])
        acc_mean, acc_std = mean_std([row["acceptance_rate"] for row in rows])
        print(
            f"| {method} | {len(rows)} | {speed_mean:.2f} ± {speed_std:.2f} | "
            f"{block_mean:.2f} ± {block_std:.2f} | {acc_mean:.3f} ± {acc_std:.3f} |"
        )


if __name__ == "__main__":
    main()
