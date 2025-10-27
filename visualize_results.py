import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _parse_out_file(out_path: Path):
    metrics_overall = {}
    metrics_events = {}
    detections = []

    stage = "header"
    with out_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if stage == "header":
                    stage = "detections"
                elif stage == "detections":
                    stage = "footer"
                continue

            if stage == "header":
                if ":" in line:
                    key, value = line.split(":", 1)
                    metrics_overall[key.strip()] = int(value.strip().replace(",", ""))
            elif stage == "detections":
                if line.startswith("#"):
                    continue
                if "," not in line:
                    continue
                pipe, timestamp = [part.strip() for part in line.split(",", 1)]
                detections.append({"pipe": pipe, "timestamp": timestamp})
            else:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metrics_events[key.strip()] = int(value.strip().replace(",", ""))

    detections_df = pd.DataFrame(detections)
    if not detections_df.empty:
        detections_df["timestamp"] = pd.to_datetime(detections_df["timestamp"])

    return metrics_overall, metrics_events, detections_df


def _plot_confusion(metrics: dict, output_path: Path, title: str):
    if not metrics:
        return

    order = [name for name in ["TP", "TN", "FP", "FN"] if name in metrics]
    values = [metrics[name] for name in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4caf50", "#2196f3", "#f44336", "#ff9800"]
    bars = ax.bar(order, values, color=colors[: len(values)])

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_detection_counts(detections_df: pd.DataFrame, output_path: Path, top_n: int = 20):
    if detections_df.empty:
        return

    counts = detections_df["pipe"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    counts.sort_values().plot(kind="barh", ax=ax, color="#607d8b")
    ax.set_xlabel("Number of detections")
    ax.set_ylabel("Pipe ID")
    ax.set_title(f"Top {len(counts)} detected pipes")
    for idx, value in enumerate(counts.sort_values()):
        ax.text(value + 0.1, idx, f"{value}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_detection_timeline(detections_df: pd.DataFrame, output_path: Path, freq: str = "7D"):
    if detections_df.empty:
        return

    timeline = (
        detections_df.set_index("timestamp")
        .sort_index()
        .resample(freq)
        .size()
        .rename("detections")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeline.index, timeline.values, marker="o", linestyle="-", color="#3f51b5")
    ax.set_xlabel(f"Date (resampled every {freq})")
    ax.set_ylabel("Detections")
    ax.set_title("Detection frequency over time")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _export_detection_tables(run_dir: Path, detections_df: pd.DataFrame):
    if detections_df.empty:
        return

    detections_df.sort_values("timestamp").to_csv(run_dir / "detections_full.csv", index=False)

    summary = (
        detections_df.groupby(detections_df["timestamp"].dt.to_period("M"))
        .size()
        .rename("detections")
        .to_frame()
    )
    summary.index = summary.index.astype(str)
    summary.to_csv(run_dir / "detections_by_month.csv")


def main():
    parser = argparse.ArgumentParser(description="Generate visual summaries from evaluation results.")
    parser.add_argument(
        "--root",
        default="results",
        type=str,
        help="Root folder that holds result subdirectories (default: results).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Results root '{root}' does not exist.")

    out_files = sorted(root.rglob("out.txt"))
    if not out_files:
        raise SystemExit(f"No out.txt files found under '{root}'.")

    for out_file in out_files:
        run_dir = out_file.parent
        layer_name = run_dir.parent.name
        run_name = run_dir.name
        print(f"Processing {layer_name}/{run_name}")

        metrics_overall, metrics_events, detections_df = _parse_out_file(out_file)

        _plot_confusion(
            metrics_overall,
            run_dir / "confusion_overall.png",
            f"Overall edge classification – {layer_name}/{run_name}",
        )
        _plot_confusion(
            metrics_events,
            run_dir / "confusion_event_level.png",
            f"Event detection summary – {layer_name}/{run_name}",
        )
        _plot_detection_counts(
            detections_df,
            run_dir / "detections_top_pipes.png",
        )
        _plot_detection_timeline(
            detections_df,
            run_dir / "detections_over_time.png",
        )
        _export_detection_tables(run_dir, detections_df)

        total_events = detections_df.shape[0]
        unique_pipes = detections_df["pipe"].nunique() if not detections_df.empty else 0
        print(
            f"  -> detections recorded: {total_events}, unique pipes: {unique_pipes}, "
            f"overall TP/TN/FP/FN: {metrics_overall}"
        )


if __name__ == "__main__":
    main()
