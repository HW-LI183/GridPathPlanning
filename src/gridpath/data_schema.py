"""环境标识、布局归属与网格序列化的稳定数据协议。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .types import Grid


def _canonical_json(payload: Any) -> bytes:
    """使用排序键、紧凑分隔符和固定字符转义生成哈希输入。"""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_hash(payload: Any, prefix: str) -> str:
    """返回带命名前缀的规范 JSON SHA256 前 24 位十六进制摘要。"""

    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _layout_payload(grid: Grid) -> dict:
    return {
        "grid_size": [grid.rows, grid.cols],
        "obstacles": [list(coord) for coord in sorted(grid.obstacles)],
    }


def layout_id(grid: Grid) -> str:
    """只根据尺寸和排序后的障碍物标识布局，不包含起终点。"""

    return canonical_hash(_layout_payload(grid), "layout")


def environment_id(grid: Grid) -> str:
    """根据布局标识、起点与终点标识一个完整任务环境。"""

    return canonical_hash(
        {"layout_id": layout_id(grid), "start": list(grid.start), "goal": list(grid.goal)},
        "environment",
    )


def layout_split(
    layout_id: str, seed: int, ratios: dict[str, int] | None = None
) -> str:
    """在生成路径前按布局确定唯一归属，默认比例为 80/10/10。

    对规范 JSON ``[seed, layout_id]`` 的完整 SHA256 整数取权重总和的模。
    桶顺序固定为 train、dev、test，其余自定义桶按名称排序，因此字典插入顺序
    不影响结果。权重允许为零，但总和必须为正整数。
    """

    if type(seed) is not int:
        raise ValueError("划分随机种子必须为整数")
    if not isinstance(layout_id, str) or not layout_id:
        raise ValueError("布局标识必须为非空字符串")
    weights = {"train": 80, "dev": 10, "test": 10} if ratios is None else dict(ratios)
    if any(not isinstance(name, str) or not name for name in weights):
        raise ValueError("划分名称必须为非空字符串")
    if any(type(weight) is not int or weight < 0 for weight in weights.values()):
        raise ValueError("划分权重必须为非负整数")
    total = sum(weights.values())
    if total == 0:
        raise ValueError("划分权重总和必须大于零")
    preferred = ("train", "dev", "test")
    names = [name for name in preferred if name in weights]
    names.extend(sorted(name for name in weights if name not in preferred))
    bucket = int.from_bytes(
        hashlib.sha256(_canonical_json([seed, layout_id])).digest(), "big"
    ) % total
    for name in names:
        if bucket < weights[name]:
            return name
        bucket -= weights[name]
    raise AssertionError("划分权重遍历未找到归属")


def grid_to_dict(grid: Grid) -> dict:
    """输出可直接写入 JSON 的网格字段，障碍坐标按行列排序。"""

    return {**_layout_payload(grid), "start": list(grid.start), "goal": list(grid.goal)}


def grid_from_record(record: Mapping) -> Grid:
    """读取直接网格字典或记录中的 ``grid`` 字典，忽略监督与参考字段。"""

    if not isinstance(record, Mapping):
        raise ValueError("网格记录必须为字典")
    payload = record.get("grid", record)
    if not isinstance(payload, Mapping):
        raise ValueError("记录中的 grid 必须为字典")
    try:
        size = payload["grid_size"]
        if (
            not isinstance(size, Sequence)
            or isinstance(size, (str, bytes))
            or len(size) != 2
        ):
            raise ValueError("grid_size 必须包含行数和列数")
        obstacles = frozenset(tuple(coord) for coord in payload["obstacles"])
        return Grid(size[0], size[1], obstacles, payload["start"], payload["goal"])
    except KeyError as exc:
        raise ValueError(f"网格记录缺少必需字段：{exc.args[0]}") from exc
    except TypeError as exc:
        raise ValueError("网格记录的坐标或障碍物格式错误") from exc
