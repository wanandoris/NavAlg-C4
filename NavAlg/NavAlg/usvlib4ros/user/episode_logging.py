import json
from datetime import datetime
from pathlib import Path


class EpisodeDataLogger:
    def __init__(self, log_dir: Path, timestamp_factory=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp_factory = timestamp_factory or self._default_timestamp
        self._file = None
        self.current_path = None

    def start_episode(self, episode_index: int) -> Path:
        self.close()
        timestamp = self._timestamp_factory()
        self.current_path = self.log_dir / f"episode_{episode_index:04d}_{timestamp}.log"
        self._file = self.current_path.open("w", encoding="utf-8")
        return self.current_path

    def log_snapshot(self, obstacle, target, ship, laser):
        if self._file is None:
            raise RuntimeError("episode log file is not open")

        payload = {
            "obstacle": obstacle,
            "target": target,
            "ship": ship,
            "laser": laser,
        }
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        if self._file is None:
            return
        self._file.close()
        self._file = None

    @staticmethod
    def _default_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
