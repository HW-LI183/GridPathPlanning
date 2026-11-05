#!/usr/bin/env python3
"""有限检查官方单目标函数；不导入或运行上游脚本的批处理入口。"""

import ast
import hashlib
import heapq
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/ppnl_upstream"
SOURCE = UPSTREAM / "ppnl-spatial-temporal-reasoning/evaluate/executor-point-sg.py"


def main():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"is_goal", "a_star", "is_optimal"}]
    namespace = {"heapq": heapq}
    # 仅执行静态审阅过的三个函数定义；不执行上游 import、文件读取或顶层调用。
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    world = [[2, 3, 0], [0, 1, 0], [0, 0, 0]]
    cases = [("right", 1), ("right unknown", 1), ("right left", 0),
             ("right left right", 1), ("down right", 2), ("left", 0),
             ("", 0), ("right\n", 0), ("right  ", 1)]
    results = []
    for actions, expected in cases:
        actual = namespace["is_goal"](world, (0, 0), actions, 3, 3)
        assert actual == expected, (actions, actual, expected)
        results.append({"actions": actions, "official_code": actual, "expected_code": expected})
    goal_empty = namespace["is_goal"](world, (0, 1), "", 3, 3)
    distance_at_goal = namespace["a_star"](world, (0, 1), (0, 1))
    assert goal_empty == 1 and distance_at_goal == 1
    report = {"upstream_commit": subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "cases": results, "empty_at_goal_code": goal_empty,
        "official_distance_at_goal": distance_at_goal,
        "note": "距离函数在目标返回1，计的是节点数；这些是官方行为探查，不是阶段2适配器实现。"}
    output = ROOT / "manifests/upstream_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
