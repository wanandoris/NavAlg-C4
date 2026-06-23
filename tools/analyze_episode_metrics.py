import csv
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOW_START = 1200
STAGE_WINDOW = 51


def _to_int(value: str) -> int:
    return int(float(value.strip()))


def _to_float(value: str) -> float:
    text = value.strip()
    return float(text) if text else 0.0


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        normalized_fieldnames = [name.strip() for name in reader.fieldnames or []]
        rows = []
        for raw_row in reader:
            row = {key.strip(): (value or "").strip() for key, value in raw_row.items()}
            rows.append(row)
    return normalized_fieldnames, rows


def read_rows_with_trimmed_headers(csv_path: Path):
    _, rows = read_rows(csv_path)
    return rows


def fix_reset_update_steps(rows):
    fixed_rows = []
    offset = 0
    prev_raw_step = None
    prev_fixed_step = None

    for row in rows:
        raw_step = _to_int(row["update_step"])

        if prev_raw_step is not None and raw_step < prev_raw_step:
            offset = prev_fixed_step or 0

        fixed_step = raw_step + offset
        fixed_row = dict(row)
        fixed_row["update_step"] = str(fixed_step)
        fixed_rows.append(fixed_row)

        prev_raw_step = raw_step
        prev_fixed_step = fixed_step

    return fixed_rows


def recompute_metrics(rows):
    total = 0
    arrived_total = 0
    collided_total = 0
    timeout_total = 0
    episode_time_total = 0.0
    steps_total = 0
    first_arrive_episode = None

    for new_episode, row in enumerate(rows):
        arrived = _to_int(row["arrived"])
        collided = _to_int(row["collided"])
        timeout = _to_int(row["timeout"])
        steps = _to_int(row["steps"])
        episode_time = _to_float(row["episode_time_sec"])

        total += 1
        arrived_total += arrived
        collided_total += collided
        timeout_total += timeout
        episode_time_total += episode_time
        steps_total += steps

        if arrived and first_arrive_episode is None:
            first_arrive_episode = new_episode

        row["episode"] = str(new_episode)
        row["first_arrive_episode"] = str(first_arrive_episode if first_arrive_episode is not None else -1)
        row["success_rate_total"] = f"{arrived_total / total:.10f}"
        row["collision_rate_total"] = f"{collided_total / total:.10f}"
        row["timeout_rate_total"] = f"{timeout_total / total:.10f}"
        row["avg_episode_time_total"] = f"{episode_time_total / total:.10f}"
        row["avg_steps_total"] = f"{steps_total / total:.10f}"

    return rows


def write_rows(fieldnames, rows, output_path: Path):
    output_fieldnames = list(fieldnames)
    if "timeout_rate_total" not in output_fieldnames:
        insert_at = output_fieldnames.index("collision_rate_total") + 1
        output_fieldnames.insert(insert_at, "timeout_rate_total")

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_overall_success_series(rows):
    success_count = 0
    x_values = []
    y_values = []

    for idx, row in enumerate(rows):
        success_count += _to_int(row["arrived"])
        x_values.append(idx)
        y_values.append(success_count / (idx + 1))

    return x_values, y_values


def build_average_return_series(rows):
    return_sum = 0.0
    x_values = []
    y_values = []

    for idx, row in enumerate(rows):
        return_sum += _to_float(row["episode_return"])
        x_values.append(idx)
        y_values.append(return_sum / (idx + 1))

    return x_values, y_values


def build_post_window_success_series(rows, start_episode: int):
    filtered = rows[start_episode:]
    success_count = 0
    x_values = []
    y_values = []

    for idx, row in enumerate(filtered, start=start_episode):
        success_count += _to_int(row["arrived"])
        x_values.append(idx)
        y_values.append(success_count / (idx - start_episode + 1))

    return x_values, y_values


def build_sliding_window_success_series(rows, start_episode: int, window_size: int):
    window = deque()
    success_sum = 0
    x_values = []
    y_values = []

    for idx, row in enumerate(rows[start_episode:], start=start_episode):
        success = _to_int(row["arrived"])
        window.append(success)
        success_sum += success

        if len(window) > window_size:
            removed = window.popleft()
            success_sum -= removed

        x_values.append(idx)
        y_values.append(success_sum / len(window))

    return x_values, y_values


def save_plot_overall_success(x_values, y_values, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, y_values, color="#1f77b4", linewidth=1.6)
    ax.set_title("Overall Success Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plot_average_return(x_values, y_values, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, y_values, color="#ff7f0e", linewidth=1.6)
    ax.set_title("Average Episode Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Return")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plot_windowed_success(
    post_window_x,
    post_window_y,
    stage_window_x,
    stage_window_y,
    output_path: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    axes[0].plot(post_window_x, post_window_y, color="#2ca02c", linewidth=1.6)
    axes[0].set_title("Overall Success Rate After Episode 1200")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(stage_window_x, stage_window_y, color="#d62728", linewidth=1.6)
    axes[1].set_title("Stage Success Rate With Sliding 1200-Episode Window")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plot_average_time_and_steps(rows, output_path: Path):
    episodes = []
    avg_times = []
    avg_steps = []

    for row in rows:
        episodes.append(_to_int(row["episode"]))
        avg_times.append(_to_float(row["avg_episode_time_total"]))
        avg_steps.append(_to_float(row["avg_steps_total"]))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(episodes, avg_times, color="#9467bd", linewidth=1.6)
    axes[0].set_title("Average Episode Time")
    axes[0].set_ylabel("Time (sec)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(episodes, avg_steps, color="#8c564b", linewidth=1.6)
    axes[1].set_title("Average Steps")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plot_summary_success_rate(rows, output_path: Path):
    update_steps = []
    success_rates = []

    for row in rows:
        update_steps.append(_to_int(row["update_step"]))
        success_rates.append(_to_float(row["success_rate_total"]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(update_steps, success_rates, color="#17becf", linewidth=1.6)
    ax.set_title("Success Rate vs PPO Update")
    ax.set_xlabel("PPO Update Step")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_single_metric_plot(rows, x_key: str, y_key: str, title: str, ylabel: str, color: str, output_path: Path):
    x_values = []
    y_values = []

    for row in rows:
        x_values.append(_to_int(row[x_key]))
        y_values.append(_to_float(row[y_key]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, y_values, color=color, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("PPO Update Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    result_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    input_csv = result_dir / "episode_metrics.csv"
    summary_csv = result_dir / "summary_metrics.csv"
    output_csv = result_dir / "episode_metrics_filtered_reindexed.csv"
    plots_dir = result_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_rows(input_csv)
    valid_rows = [row for row in rows if _to_int(row["steps"]) >= 5]
    valid_rows = recompute_metrics(valid_rows)
    write_rows(fieldnames, valid_rows, output_csv)

    overall_x, overall_y = build_overall_success_series(valid_rows)
    average_return_x, average_return_y = build_average_return_series(valid_rows)
    post_window_x, post_window_y = build_post_window_success_series(valid_rows, WINDOW_START)
    stage_window_x, stage_window_y = build_sliding_window_success_series(
        valid_rows,
        WINDOW_START,
        STAGE_WINDOW,
    )

    save_plot_overall_success(overall_x, overall_y, plots_dir / "overall_success_rate.png")
    save_plot_average_return(
        average_return_x,
        average_return_y,
        plots_dir / "average_episode_return.png",
    )
    save_plot_windowed_success(
        post_window_x,
        post_window_y,
        stage_window_x,
        stage_window_y,
        plots_dir / "success_rate_after_1200.png",
    )
    save_plot_average_time_and_steps(
        valid_rows,
        plots_dir / "average_time_and_steps.png",
    )

    if summary_csv.exists():
        summary_rows = fix_reset_update_steps(read_rows_with_trimmed_headers(summary_csv))
        save_plot_summary_success_rate(
            summary_rows,
            plots_dir / "summary_success_rate_vs_ppo_update.png",
        )
        save_single_metric_plot(
            summary_rows,
            "update_step",
            "actor_loss",
            "Actor Loss vs PPO Update",
            "Actor Loss",
            "#1f77b4",
            plots_dir / "actor_loss_vs_ppo_update.png",
        )
        save_single_metric_plot(
            summary_rows,
            "update_step",
            "critic_loss",
            "Critic Loss vs PPO Update",
            "Critic Loss",
            "#d62728",
            plots_dir / "critic_loss_vs_ppo_update.png",
        )
        save_single_metric_plot(
            summary_rows,
            "update_step",
            "entropy",
            "Entropy vs PPO Update",
            "Entropy",
            "#2ca02c",
            plots_dir / "entropy_vs_ppo_update.png",
        )

    print(f"Input rows: {len(rows)}")
    print(f"Filtered rows kept: {len(valid_rows)}")
    print(f"Removed rows with steps < 5: {len(rows) - len(valid_rows)}")
    print(f"Filtered CSV saved to: {output_csv}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
