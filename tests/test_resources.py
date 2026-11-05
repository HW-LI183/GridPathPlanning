"""预算账本必须保留失败成本并阻止并发、未结账和超预算运行。"""
from datetime import datetime, timedelta, timezone

import pytest

from gridpath.resources import GPUBudget, close_interrupted
from gridpath.training_config import read_json, write_json


def test_budget_records_success_and_failure_and_rejects_over_cap(tmp_path):
    path = tmp_path / "budget.json"
    with GPUBudget(path, 1, 60, "success") as budget:
        budget.check()
    assert read_json(path)["entries"][0]["status"] == "complete"
    with pytest.raises(TimeoutError):
        with GPUBudget(path, 1, 60, "failure") as budget:
            budget.check(reserve_seconds=120)
    data = read_json(path)
    assert data["entries"][1]["status"] == "failed"
    data["entries"][0]["wall_seconds"] = 3600
    write_json(path, data)
    with pytest.raises(RuntimeError, match="用尽"):
        with GPUBudget(path, 1, 60, "over-cap"):
            pass


def test_unclosed_budget_requires_explicit_conservative_accounting(tmp_path):
    path = tmp_path / "budget.json"
    write_json(path, {"entries": [{"id": "interrupted", "status": "running", "wall_seconds": 0,
                                   "started_at": (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()}]})
    with pytest.raises(RuntimeError, match="未结束"):
        with GPUBudget(path, 1, 60, "next"):
            pass
    result = close_interrupted(path, "interrupted")
    assert result["status"] == "interrupted" and result["wall_seconds"] >= 5
    with GPUBudget(path, 1, 60, "next"):
        pass


def test_budget_blocks_concurrent_process_and_manual_close(tmp_path):
    path = tmp_path / "budget.json"
    with GPUBudget(path, 1, 60, "active") as budget:
        with pytest.raises(BlockingIOError):
            with GPUBudget(path, 1, 60, "duplicate"):
                pass
        with pytest.raises(BlockingIOError):
            close_interrupted(path, budget.entry["id"])
