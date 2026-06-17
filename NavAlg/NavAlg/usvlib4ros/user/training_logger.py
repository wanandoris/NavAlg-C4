import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_LIVE_PLOT_ENV = os.environ.get("USV_TRAINING_LIVE_PLOT", "0") == "1"

try:
    import matplotlib
    if _LIVE_PLOT_ENV:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")
            _LIVE_PLOT_ENV = False
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if _LIVE_PLOT_ENV:
        plt.ion()
except Exception:  # pragma: no cover
    plt = None
    _LIVE_PLOT_ENV = False

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class EpisodeMetrics:
    episode: int
    episode_return: float
    steps: int
    episode_time_sec: float
    arrived: bool
    collided: bool
    timeout: bool
    actor_loss: float | None
    critic_loss: float | None
    total_loss: float | None
    entropy: float | None
    auc_return: float
    first_arrive_episode: int | None


class TrainingLogger:
    """轻量训练日志器，同时记录 TensorBoard 和 CSV。"""

    def __init__(self, root_dir: str | Path = "Results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root_dir) / f"ppo_nav_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.run_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.live_plot_enabled = _LIVE_PLOT_ENV and plt is not None

        self.episode_csv_path = self.run_dir / "episode_metrics.csv"
        self.summary_csv_path = self.run_dir / "summary_metrics.csv"
        self.update_csv_path = self.run_dir / "ppo_update_metrics.csv"

        self.writer = SummaryWriter(log_dir=str(self.run_dir)) if SummaryWriter else None

        self.total_episodes = 0
        self.arrive_count = 0
        self.collision_count = 0
        self.total_episode_time = 0.0
        self.total_steps = 0
        self.return_auc = 0.0
        self.first_arrive_episode = None
        self._episode_rows: list[dict[str, Any]] = []
        self._summary_rows: list[dict[str, Any]] = []
        self._update_rows: list[dict[str, Any]] = []
        self._live_figures: dict[str, tuple[Any, Any]] = {}

        self._init_csv_files()
        if self.live_plot_enabled:
            self._init_live_windows()

    def _init_csv_files(self):
        self._write_csv_header(
            self.episode_csv_path,
            [
                "episode",
                "episode_return",
                "steps",
                "episode_time_sec",
                "arrived",
                "collided",
                "timeout",
                "actor_loss",
                "critic_loss",
                "total_loss",
                "entropy",
                "auc_return",
                "first_arrive_episode",
                "success_rate_total",
                "collision_rate_total",
                "avg_episode_time_total",
                "avg_steps_total",
            ],
        )
        self._write_csv_header(
            self.summary_csv_path,
            [
                "update_step",
                "buffer_size",
                "actor_loss",
                "critic_loss",
                "total_loss",
                "entropy",
                "total_episodes",
                "success_rate_total",
                "collision_rate_total",
                "avg_episode_time_total",
                "avg_steps_total",
                "auc_return",
                "first_arrive_episode",
            ],
        )
        self._write_csv_header(
            self.update_csv_path,
            [
                "update_step",
                "buffer_size",
                "actor_loss",
                "critic_loss",
                "total_loss",
                "entropy",
            ],
        )

    @staticmethod
    def _write_csv_header(path: Path, header: list[str]):
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    @staticmethod
    def _append_csv_row(path: Path, row: list[Any]):
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def log_update(self, global_step: int, update_metrics: dict[str, float] | None):
        if not update_metrics:
            return

        row_dict = {
            "update_step": global_step,
            "buffer_size": update_metrics.get("buffer_size"),
            "actor_loss": update_metrics.get("actor_loss"),
            "critic_loss": update_metrics.get("critic_loss"),
            "total_loss": update_metrics.get("total_loss"),
            "entropy": update_metrics.get("entropy"),
        }
        row = [
            row_dict["update_step"],
            row_dict["buffer_size"],
            row_dict["actor_loss"],
            row_dict["critic_loss"],
            row_dict["total_loss"],
            row_dict["entropy"],
        ]
        self._update_rows.append(row_dict)
        self._append_csv_row(self.update_csv_path, row)

        summary_row = {
            "update_step": global_step,
            "buffer_size": update_metrics.get("buffer_size"),
            "actor_loss": update_metrics.get("actor_loss"),
            "critic_loss": update_metrics.get("critic_loss"),
            "total_loss": update_metrics.get("total_loss"),
            "entropy": update_metrics.get("entropy"),
            "total_episodes": self.total_episodes,
            "success_rate_total": self.arrive_count / self.total_episodes if self.total_episodes else 0.0,
            "collision_rate_total": self.collision_count / self.total_episodes if self.total_episodes else 0.0,
            "avg_episode_time_total": self.total_episode_time / self.total_episodes if self.total_episodes else 0.0,
            "avg_steps_total": self.total_steps / self.total_episodes if self.total_episodes else 0.0,
            "auc_return": self.return_auc,
            "first_arrive_episode": self.first_arrive_episode,
        }
        self._summary_rows.append(summary_row)
        self._append_csv_row(
            self.summary_csv_path,
            [
                summary_row["update_step"],
                summary_row["buffer_size"],
                summary_row["actor_loss"],
                summary_row["critic_loss"],
                summary_row["total_loss"],
                summary_row["entropy"],
                summary_row["total_episodes"],
                summary_row["success_rate_total"],
                summary_row["collision_rate_total"],
                summary_row["avg_episode_time_total"],
                summary_row["avg_steps_total"],
                summary_row["auc_return"],
                summary_row["first_arrive_episode"],
            ],
        )

        if self.writer:
            self.writer.add_scalar("loss/actor", update_metrics["actor_loss"], global_step)
            self.writer.add_scalar("loss/critic", update_metrics["critic_loss"], global_step)
            self.writer.add_scalar("loss/total", update_metrics["total_loss"], global_step)
            self.writer.add_scalar("policy/entropy", update_metrics["entropy"], global_step)

        self._plot_update_metrics()
        self._plot_summary_metrics()
        self._refresh_live_windows()

    def log_episode(
        self,
        episode: int,
        episode_return: float,
        steps: int,
        episode_time_sec: float,
        arrived: bool,
        collided: bool,
        timeout: bool,
        last_update_metrics: dict[str, float] | None,
    ) -> EpisodeMetrics:
        self.total_episodes += 1
        self.arrive_count += int(arrived)
        self.collision_count += int(collided)
        self.total_episode_time += episode_time_sec
        self.total_steps += steps
        self.return_auc += episode_return

        if arrived and self.first_arrive_episode is None:
            self.first_arrive_episode = episode

        metrics = EpisodeMetrics(
            episode=episode,
            episode_return=episode_return,
            steps=steps,
            episode_time_sec=episode_time_sec,
            arrived=arrived,
            collided=collided,
            timeout=timeout,
            actor_loss=self._metric_value(last_update_metrics, "actor_loss"),
            critic_loss=self._metric_value(last_update_metrics, "critic_loss"),
            total_loss=self._metric_value(last_update_metrics, "total_loss"),
            entropy=self._metric_value(last_update_metrics, "entropy"),
            auc_return=self.return_auc,
            first_arrive_episode=self.first_arrive_episode,
        )

        total_success_rate = self.arrive_count / self.total_episodes
        total_collision_rate = self.collision_count / self.total_episodes
        total_avg_time = self.total_episode_time / self.total_episodes
        total_avg_steps = self.total_steps / self.total_episodes
        row_dict = {
            "episode": metrics.episode,
            "episode_return": metrics.episode_return,
            "steps": metrics.steps,
            "episode_time_sec": metrics.episode_time_sec,
            "arrived": int(metrics.arrived),
            "collided": int(metrics.collided),
            "timeout": int(metrics.timeout),
            "actor_loss": metrics.actor_loss,
            "critic_loss": metrics.critic_loss,
            "total_loss": metrics.total_loss,
            "entropy": metrics.entropy,
            "auc_return": metrics.auc_return,
            "first_arrive_episode": metrics.first_arrive_episode,
            "success_rate_total": total_success_rate,
            "collision_rate_total": total_collision_rate,
            "avg_episode_time_total": total_avg_time,
            "avg_steps_total": total_avg_steps,
        }
        self._episode_rows.append(row_dict)

        self._append_csv_row(
            self.episode_csv_path,
            [
                row_dict["episode"],
                row_dict["episode_return"],
                row_dict["steps"],
                row_dict["episode_time_sec"],
                row_dict["arrived"],
                row_dict["collided"],
                row_dict["timeout"],
                row_dict["actor_loss"],
                row_dict["critic_loss"],
                row_dict["total_loss"],
                row_dict["entropy"],
                row_dict["auc_return"],
                row_dict["first_arrive_episode"],
                row_dict["success_rate_total"],
                row_dict["collision_rate_total"],
                row_dict["avg_episode_time_total"],
                row_dict["avg_steps_total"],
            ],
        )

        if self.writer:
            self.writer.add_scalar("episode/return", episode_return, episode)
            self.writer.add_scalar("episode/steps", steps, episode)
            self.writer.add_scalar("episode/time_sec", episode_time_sec, episode)
            self.writer.add_scalar("episode/success", int(arrived), episode)
            self.writer.add_scalar("episode/collision", int(collided), episode)
            self.writer.add_scalar("metrics/auc_return", self.return_auc, episode)
            self.writer.add_scalar("metrics/success_rate_total", total_success_rate, episode)
            self.writer.add_scalar("metrics/collision_rate_total", total_collision_rate, episode)
            self.writer.add_scalar("metrics/avg_episode_time_total", total_avg_time, episode)
            self.writer.add_scalar("metrics/avg_steps_total", total_avg_steps, episode)

        self._plot_episode_metrics()
        return metrics

    def close(self):
        self._plot_all()
        self._refresh_live_windows(force_draw=True)
        if self.writer:
            self.writer.flush()
            self.writer.close()

    @staticmethod
    def _metric_value(metrics: dict[str, float] | None, key: str) -> float | None:
        if not metrics:
            return None
        return metrics.get(key)

    def _plot_all(self):
        self._plot_episode_metrics()
        self._plot_update_metrics()
        self._plot_summary_metrics()
        self._refresh_live_windows()

    def _plot_episode_metrics(self):
        if plt is None or not self._episode_rows:
            return

        episodes = [row["episode"] for row in self._episode_rows]
        self._save_line_plot(
            self.plot_dir / "episode_return.png",
            episodes,
            [row["episode_return"] for row in self._episode_rows],
            "Episode Return",
            "Episode",
            "Return",
        )
        self._save_line_plot(
            self.plot_dir / "episode_steps.png",
            episodes,
            [row["steps"] for row in self._episode_rows],
            "Episode Steps",
            "Episode",
            "Steps",
        )
        self._save_line_plot(
            self.plot_dir / "episode_time.png",
            episodes,
            [row["episode_time_sec"] for row in self._episode_rows],
            "Episode Time",
            "Episode",
            "Seconds",
        )
        self._save_multi_line_plot(
            self.plot_dir / "success_collision_total.png",
            episodes,
            [
                ("Success Rate Total", [row["success_rate_total"] for row in self._episode_rows]),
                ("Collision Rate Total", [row["collision_rate_total"] for row in self._episode_rows]),
            ],
            "Success / Collision Rate",
            "Episode",
            "Rate",
        )
        self._save_line_plot(
            self.plot_dir / "auc_return.png",
            episodes,
            [row["auc_return"] for row in self._episode_rows],
            "AUC Return",
            "Episode",
            "AUC",
        )

    def _plot_update_metrics(self):
        if plt is None or not self._update_rows:
            return

        steps = [row["update_step"] for row in self._update_rows]
        self._save_multi_line_plot(
            self.plot_dir / "ppo_losses.png",
            steps,
            [
                ("Actor Loss", [row["actor_loss"] for row in self._update_rows]),
                ("Critic Loss", [row["critic_loss"] for row in self._update_rows]),
                ("Total Loss", [row["total_loss"] for row in self._update_rows]),
            ],
            "PPO Loss Curves",
            "Update Step",
            "Loss",
        )
        self._save_line_plot(
            self.plot_dir / "entropy.png",
            steps,
            [row["entropy"] for row in self._update_rows],
            "Policy Entropy",
            "Update Step",
            "Entropy",
        )
        self._save_line_plot(
            self.plot_dir / "buffer_size.png",
            steps,
            [row["buffer_size"] for row in self._update_rows],
            "Buffer Size",
            "Update Step",
            "Samples",
        )

    def _plot_summary_metrics(self):
        if plt is None or not self._summary_rows:
            return

        updates = [row["update_step"] for row in self._summary_rows]
        self._save_multi_line_plot(
            self.plot_dir / "summary_rates.png",
            updates,
            [
                ("Success Rate Total", [row["success_rate_total"] for row in self._summary_rows]),
                ("Collision Rate Total", [row["collision_rate_total"] for row in self._summary_rows]),
            ],
            "Summary Metrics",
            "Update Step",
            "Rate",
        )
        self._save_multi_line_plot(
            self.plot_dir / "summary_efficiency.png",
            updates,
            [
                ("Avg Episode Time Total", [row["avg_episode_time_total"] for row in self._summary_rows]),
                ("Avg Steps Total", [row["avg_steps_total"] for row in self._summary_rows]),
                ("AUC Return", [row["auc_return"] for row in self._summary_rows]),
            ],
            "Summary Efficiency",
            "Update Step",
            "Value",
        )

    def _save_line_plot(
        self,
        path: Path,
        x_values: list[float],
        y_values: list[float | None],
        title: str,
        x_label: str,
        y_label: str,
    ):
        series = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
        if not series:
            return
        xs = [x for x, _ in series]
        ys = [y for _, y in series]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _save_multi_line_plot(
        self,
        path: Path,
        x_values: list[float],
        series_list: list[tuple[str, list[float | None]]],
        title: str,
        x_label: str,
        y_label: str,
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        has_data = False

        for label, y_values in series_list:
            series = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
            if not series:
                continue
            has_data = True
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            ax.plot(xs, ys, linewidth=1.8, label=label)

        if not has_data:
            plt.close(fig)
            return

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _init_live_windows(self):
        if plt is None:
            return

        self._live_figures["episode"] = plt.subplots(2, 2, figsize=(11, 7))
        self._live_figures["update"] = plt.subplots(2, 2, figsize=(11, 7))
        self._live_figures["summary"] = plt.subplots(1, 2, figsize=(11, 4.5))

        for key, (fig, axes) in self._live_figures.items():
            try:
                fig.canvas.manager.set_window_title(f"USV Training - {key}")
            except Exception:
                pass
            fig.tight_layout()
            fig.show()

    def _refresh_live_windows(self, force_draw: bool = False):
        if not self.live_plot_enabled or plt is None:
            return

        self._draw_live_episode_window()
        self._draw_live_update_window()
        self._draw_live_summary_window()

        if force_draw:
            plt.pause(0.001)
        else:
            plt.pause(0.0001)

    def _draw_live_episode_window(self):
        if "episode" not in self._live_figures or not self._episode_rows:
            return
        fig, axes = self._live_figures["episode"]
        for ax in axes.flat:
            ax.clear()

        episodes = [row["episode"] for row in self._episode_rows]
        axes[0, 0].plot(episodes, [row["episode_return"] for row in self._episode_rows], color="tab:blue")
        axes[0, 0].set_title("Episode Return")
        axes[0, 1].plot(episodes, [row["steps"] for row in self._episode_rows], color="tab:orange")
        axes[0, 1].set_title("Episode Steps")
        axes[1, 0].plot(episodes, [row["episode_time_sec"] for row in self._episode_rows], color="tab:green")
        axes[1, 0].set_title("Episode Time")
        axes[1, 1].plot(episodes, [row["success_rate_total"] for row in self._episode_rows], label="Success", color="tab:green")
        axes[1, 1].plot(episodes, [row["collision_rate_total"] for row in self._episode_rows], label="Collision", color="tab:red")
        axes[1, 1].set_title("Success / Collision")
        axes[1, 1].legend()
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def _draw_live_update_window(self):
        if "update" not in self._live_figures or not self._update_rows:
            return
        fig, axes = self._live_figures["update"]
        for ax in axes.flat:
            ax.clear()

        steps = [row["update_step"] for row in self._update_rows]
        axes[0, 0].plot(steps, [row["actor_loss"] for row in self._update_rows], label="Actor")
        axes[0, 0].plot(steps, [row["critic_loss"] for row in self._update_rows], label="Critic")
        axes[0, 0].plot(steps, [row["total_loss"] for row in self._update_rows], label="Total")
        axes[0, 0].set_title("PPO Loss")
        axes[0, 0].legend()
        axes[0, 1].plot(steps, [row["entropy"] for row in self._update_rows], color="tab:purple")
        axes[0, 1].set_title("Entropy")
        axes[1, 0].plot(steps, [row["buffer_size"] for row in self._update_rows], color="tab:brown")
        axes[1, 0].set_title("Buffer Size")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        axes[1, 1].axis("off")
        fig.tight_layout()
        fig.canvas.draw_idle()

    def _draw_live_summary_window(self):
        if "summary" not in self._live_figures or not self._summary_rows:
            return
        fig, axes = self._live_figures["summary"]
        for ax in axes.flat:
            ax.clear()

        updates = [row["update_step"] for row in self._summary_rows]
        axes[0].plot(updates, [row["success_rate_total"] for row in self._summary_rows], label="Success")
        axes[0].plot(updates, [row["collision_rate_total"] for row in self._summary_rows], label="Collision")
        axes[0].set_title("Summary Rates")
        axes[0].legend()
        axes[1].plot(updates, [row["avg_episode_time_total"] for row in self._summary_rows], label="Time")
        axes[1].plot(updates, [row["avg_steps_total"] for row in self._summary_rows], label="Steps")
        axes[1].plot(updates, [row["auc_return"] for row in self._summary_rows], label="AUC")
        axes[1].set_title("Summary Efficiency")
        axes[1].legend()
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw_idle()
