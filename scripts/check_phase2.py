#!/usr/bin/env python3
"""执行有固定上限的阶段 2 CPU 检查，不构造正式数据、不训练或下载。"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridpath.bfs import shortest_path
from gridpath.executor import execute
from gridpath.metrics import (aggregate_path_metrics, aggregate_repair_metrics,
                              aggregate_verification_metrics)
from gridpath.official import load_official_is_goal
from gridpath.types import ACTION_ORDER, EXECUTION_PROTOCOL_VERSION, ErrorReason, Grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-cases", type=int, default=512, help="0到4096条短随机路径")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/phase2_checks.json")
    args = parser.parse_args()
    if not 0 <= args.random_cases <= 4096:
        parser.error("短检查限制为 0..4096 条；正式数据构造属于阶段 3")
    started = time.perf_counter()
    project = json.loads((ROOT / "configs/project.json").read_text())
    probe = json.loads((ROOT / "manifests/upstream_probe.json").read_text())
    seed = project["seed"] if args.seed is None else args.seed
    rng = random.Random(seed)
    upstream = ROOT / "third_party/ppnl_upstream"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != project["upstream_commit"] or probe["upstream_commit"] != commit:
        raise ValueError("阶段 1 记录、项目配置和上游 HEAD 不一致")
    official = load_official_is_goal(ROOT / probe["source"], expected_sha256=probe["source_sha256"])
    counts = Counter()

    def compare(grid, plan):
        result = execute(grid, plan)
        if not result.actions or any(action not in ACTION_ORDER for action in result.actions):
            raise ValueError("官方差分仅接受规范化后非空且合法的动作序列")
        observed = official(grid.to_world(), grid.start, " ".join(result.actions), grid.rows, grid.cols)
        expected = 1 if result.success else 2 if result.error_reason == ErrorReason.OBSTACLE_COLLISION else 0
        if observed != expected:
            raise AssertionError({"grid": grid.to_world(), "plan": plan,
                                  "official": observed, "strict_result": result.to_dict()})
        counts["total"] += 1
        counts[f"official_code_{observed}"] += 1

    base = Grid.from_world([[2, 3, 0], [0, 1, 0], [0, 0, 0]])
    for plan in ("right", "right left", "right left right", "down right", "left", "right up"):
        compare(base, plan)

    reference_checks = 0
    all_cells = [(row, col) for row in range(6) for col in range(6)]
    for _ in range(args.random_cases):
        start, goal = rng.sample(all_cells, 2)
        available = [cell for cell in all_cells if cell not in (start, goal)]
        grid = Grid(6, 6, frozenset(rng.sample(available, rng.randint(1, 5))), start, goal)
        plan = tuple(rng.choice(ACTION_ORDER) for _ in range(rng.randint(1, 24)))
        compare(grid, plan)
        shortest = shortest_path(grid)
        if shortest.reachable:
            # 用独立 BFS 路径保证差分检查也覆盖足够多的成功候选。
            compare(grid, shortest.actions)
            result = execute(grid, shortest.actions)
            if not result.success or result.executed_steps != shortest.distance:
                raise AssertionError("BFS 路径无法成功执行或距离不一致")
            reference_checks += 1

    demo_plans = ["right", "right left right", "down right", "left", "jump", "down", ""]
    demo_results = [execute(base, plan) for plan in demo_plans]
    demo_predictions = ["valid", "valid", "valid", "invalid | step=1 | reason=out_of_bounds",
                        "invalid | step=1 | reason=invalid_action_token", "invalid", "valid"]
    fixed = execute(base, "right")
    repaired = [fixed if index in (2, 4) else result for index, result in enumerate(demo_results)]
    differences = []
    for plan in ("right jump", "RIGHT", "right\n", "rightrightleft"):
        strict = execute(base, plan)
        differences.append({"raw_plan": plan,
                            "official_code_on_raw_text": official(base.to_world(), base.start, plan, 3, 3),
                            "strict_label": strict.verification_label})

    report = {
        "phase": 2, "status": "complete", "checked_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": EXECUTION_PROTOCOL_VERSION, "python_executable": sys.executable,
        "seed": seed, "upstream_commit": commit, "upstream_source_sha256": probe["source_sha256"],
        "project_file_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*ROOT.glob("src/gridpath/*.py"), Path(__file__).resolve(),
                                ROOT / "configs/project.json", ROOT / "pyproject.toml"])
        },
        "random_path_count": args.random_cases,
        "official_differential": {**counts, "mismatch_count": 0},
        "bfs_successful_reference_checks": reference_checks,
        "intentional_raw_text_differences": differences,
        "demonstration_scope": "仅人工样例的评分接口检查，不是模型实验结果或正式数据集",
        "demo_grid": base.to_world(),
        "demo_cases": [{"raw_plan": plan, **result.to_dict()} for plan, result in zip(demo_plans, demo_results)],
        "demo_path_metrics": aggregate_path_metrics(demo_results, [1] * len(demo_results)),
        "demo_verification_metrics": aggregate_verification_metrics(demo_results, demo_predictions),
        "demo_repair_metrics": aggregate_repair_metrics(demo_results, repaired),
        "resources": {"gpu_hours": 0, "training_runs": 0, "downloads": 0,
                      "formal_dataset_samples_created": 0, "cpu_wall_seconds": round(time.perf_counter() - started, 6)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "official_differential": report["official_differential"],
                      "bfs_reference_checks": reference_checks, "resources": report["resources"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
