"""仅为离线一致性检查加载固定上游函数，不运行其顶层批处理。"""

import ast
import hashlib
from collections.abc import Callable
from pathlib import Path


def load_official_is_goal(source: Path, *, expected_sha256: str) -> Callable:
    """调用者显式提供阶段 1 核对的哈希，防止未经审阅的源码漂移。"""
    content = Path(source).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("官方执行器源码哈希不符，请先复核上游版本")
    parsed = ast.parse(content, filename=str(source))
    selected = [node for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name == "is_goal"]
    if len(selected) != 1 or selected[0].decorator_list:
        raise ValueError("上游没有唯一的预期 is_goal 函数")
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["is_goal"]
