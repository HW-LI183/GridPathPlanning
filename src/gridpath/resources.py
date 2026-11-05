"""单 GPU 共用的保守 wall-time 账本和会话时间上限。"""

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import time
import uuid

from .training_config import read_json, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GPUBudget(AbstractContextManager):
    def __init__(self, path: Path, cap_hours: float, session_seconds: float, label: str):
        self.path, self.cap_seconds = path, cap_hours * 3600
        self.session_seconds, self.label = session_seconds, label

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.data = read_json(self.path) if self.path.exists() else {
                "schema_version": "gridpath.gpu_budget.v1", "entries": []}
            if any(e["status"] == "running" for e in self.data["entries"]):
                raise RuntimeError("上次会话未结束；确认进程停止后先用 close_interrupted_run.py 结账")
            used = sum(e["wall_seconds"] for e in self.data["entries"])
            remaining = self.cap_seconds - used
            if remaining <= 0:
                raise RuntimeError("全项目 GPU 时间预算已用尽")
            self.started = time.monotonic()
            self.deadline = self.started + min(self.session_seconds, remaining)
            self.entry = {"id": uuid.uuid4().hex, "label": self.label,
                          "status": "running", "started_at": utc_now(), "wall_seconds": 0}
            self.data["entries"].append(self.entry)
            write_json(self.path, self.data)
            return self
        except BaseException:
            self.lock.close()
            raise

    def check(self, reserve_seconds: float = 0):
        if time.monotonic() + reserve_seconds >= self.deadline:
            raise TimeoutError("会话或全项目 GPU wall-time 上限将到达，请从已提交的 checkpoint 继续")

    @property
    def remaining_seconds(self):
        return max(0, self.deadline - time.monotonic())

    def __exit__(self, exc_type, exc, tb):
        self.entry.update(status="complete" if exc_type is None else "failed",
                          finished_at=utc_now(), wall_seconds=time.monotonic() - self.started)
        write_json(self.path, self.data)
        self.lock.close()
        return False


def close_interrupted(path: Path, entry_id: str) -> dict:
    """调用者必须已确认旧进程停止；锁确保不与活跃的本项目会话冲突。"""
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = read_json(path)
        entry = next(e for e in data["entries"] if e["id"] == entry_id)
        if entry["status"] != "running":
            raise ValueError("只能关闭未正常结束的会话")
        now = datetime.now(timezone.utc)
        entry.update(status="interrupted", finished_at=now.isoformat(),
                     wall_seconds=max(0, (now - datetime.fromisoformat(entry["started_at"])).total_seconds()),
                     accounting="从启动到人工结账的保守上界")
        write_json(path, data)
        return entry
