from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def build_tensorboard_log_dir(base_dir="runs", timestamp=None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir) / f"ppo_nav_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


class TensorBoardMetricsWriter:
    def __init__(self, writer=None, log_dir=None):
        if writer is None:
            log_dir = Path(log_dir) if log_dir is not None else build_tensorboard_log_dir()
            writer = SummaryWriter(log_dir=str(log_dir))
        self.writer = writer

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, float(value), int(step))

    def flush(self):
        flush = getattr(self.writer, "flush", None)
        if flush is not None:
            flush()

    def close(self):
        close = getattr(self.writer, "close", None)
        if close is not None:
            close()

    def log_update(self, step, total_loss, policy_loss, value_loss, entropy, ratio_mean, advantage_mean):
        self.add_scalar("loss/total", total_loss, step)
        self.add_scalar("loss/policy", policy_loss, step)
        self.add_scalar("loss/value", value_loss, step)
        self.add_scalar("policy/entropy", entropy, step)
        self.add_scalar("policy/ratio_mean", ratio_mean, step)
        self.add_scalar("policy/advantage_mean", advantage_mean, step)
        self.flush()

    def log_episode(self, episode, reward, length):
        self.add_scalar("reward/episode_total", reward, episode)
        self.add_scalar("episode/length", length, episode)
        self.flush()
