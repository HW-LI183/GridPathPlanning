"""在原路径坐标系中施加受控错误，并保存可复算的反事实编辑。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import random

from .executor import execute
from .types import ACTION_DELTAS, ACTION_ORDER, ErrorReason, Grid


EDIT_KINDS = ("substitute", "delete", "insert", "truncate", "invalid_token")
INVALID_TOKEN = "<invalid>"


@dataclass(frozen=True)
class Edit:
    """index 总是引用原路径；插入发生于该位置之前，截断保留此前动作。"""

    kind: str
    index: int
    token: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in EDIT_KINDS:
            raise ValueError(f"未知编辑类型：{self.kind}")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("编辑索引必须为非负整数")
        if self.kind in ("delete", "truncate"):
            if self.token is not None:
                raise ValueError("删除与截断编辑不能携带 token")
        elif (
            not isinstance(self.token, str)
            or not self.token
            or self.token.split() != [self.token]
            or self.token.lower() != self.token
        ):
            raise ValueError("编辑 token 必须为单个非空小写 token")
        elif self.kind == "invalid_token":
            if self.token in ACTION_DELTAS:
                raise ValueError("invalid_token 必须使用四方向之外的 token")
        elif self.token not in ACTION_DELTAS:
            raise ValueError("替换或插入必须使用规范四方向 token")


@dataclass(frozen=True)
class Corruption:
    """保存联合候选、原索引编辑，以及每次编辑单独作用后的候选。"""

    actions: tuple[str, ...]
    edits: tuple[Edit, ...]
    single_actions: tuple[tuple[str, ...], ...]


def edit_to_dict(edit: Edit) -> dict[str, str | int | None]:
    """序列化完整编辑日志，包括无 token 操作的空值。"""

    return {"kind": edit.kind, "index": edit.index, "token": edit.token}


def edit_from_dict(value: Mapping) -> Edit:
    """严格读取编辑日志，避免额外字段掩盖错误协议。"""

    if not isinstance(value, Mapping) or not {"kind", "index"} <= value.keys():
        raise ValueError("编辑日志必须包含 kind 和 index")
    if set(value) - {"kind", "index", "token"}:
        raise ValueError("编辑日志含有未知字段")
    return Edit(value["kind"], value["index"], value.get("token"))


def _base_actions(base: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(base, Sequence) or isinstance(base, (str, bytes, bytearray)):
        raise TypeError("base 必须为规范动作 token 序列，而不是路径文本")
    actions = tuple(base)
    if any(not isinstance(action, str) or action not in ACTION_DELTAS for action in actions):
        raise ValueError("base 只能包含规范的小写四方向 token")
    return actions


def apply_edits(base: Sequence[str], edits: Sequence[Edit]) -> tuple[str, ...]:
    """统一按原索引编辑；拒绝同位置编辑及包含截断的多编辑组合。"""

    actions = _base_actions(base)
    if not isinstance(edits, Sequence) or any(not isinstance(edit, Edit) for edit in edits):
        raise TypeError("edits 必须为 Edit 序列")
    edits = tuple(edits)
    if len({edit.index for edit in edits}) != len(edits):
        raise ValueError("多个编辑不能使用同一原路径索引")
    if len(edits) > 1 and any(edit.kind == "truncate" for edit in edits):
        raise ValueError("多编辑组合不允许截断，以免覆盖或丢弃另一编辑")

    for edit in edits:
        upper_bound = len(actions) if edit.kind == "insert" else len(actions) - 1
        if edit.index > upper_bound:
            raise ValueError("编辑索引超出原路径可编辑范围")
        if edit.kind == "substitute" and edit.token == actions[edit.index]:
            raise ValueError("替换 token 不能与原动作相同")
        if edit.kind == "truncate":
            return actions[:edit.index]

    by_index = {edit.index: edit for edit in edits}
    output: list[str] = []
    for index in range(len(actions) + 1):
        edit = by_index.get(index)
        if edit is not None and edit.kind == "insert":
            output.append(edit.token)
        if index == len(actions):
            break
        if edit is None or edit.kind == "insert":
            output.append(actions[index])
        elif edit.kind in ("substitute", "invalid_token"):
            output.append(edit.token)
        # delete 不输出原动作，其后索引仍引用未修改的原序列。
    return tuple(output)


def _validate_request(grid: Grid, base: Sequence[str], max_actions: int) -> tuple[str, ...]:
    if type(max_actions) is not int or max_actions < 0:
        raise ValueError("max_actions 必须为非负整数")
    actions = _base_actions(base)
    if len(actions) > max_actions:
        raise ValueError("原路径已超过 max_actions 限制")
    if not execute(grid, actions).success:
        raise ValueError("受控错误只能注入执行成功的原路径")
    return actions


@lru_cache(maxsize=64)
def _single_pool(grid: Grid, base: tuple[str, ...], max_actions: int) -> tuple[Corruption, ...]:
    """缓存确定性结果，不缓存或消费随机状态；理论候选数不超过 10L+4。"""

    edits: list[Edit] = []
    for index, original in enumerate(base):
        edits.extend(Edit("substitute", index, token) for token in ACTION_ORDER if token != original)
    edits.extend(Edit("delete", index) for index in range(len(base)))
    if len(base) < max_actions:
        edits.extend(
            Edit("insert", index, token)
            for index in range(len(base) + 1)
            for token in ACTION_ORDER
        )
    edits.extend(Edit("truncate", index) for index in range(len(base)))
    edits.extend(Edit("invalid_token", index, INVALID_TOKEN) for index in range(len(base)))

    seen: set[tuple[str, ...]] = set()
    pool: list[Corruption] = []
    for edit in edits:
        candidate = apply_edits(base, (edit,))
        if candidate in seen:
            continue
        seen.add(candidate)
        if not execute(grid, candidate).success:
            pool.append(Corruption(candidate, (edit,), (candidate,)))
    return tuple(pool)


def enumerate_single_corruptions(
    grid: Grid, base: Sequence[str], max_actions: int = 72
) -> list[Corruption]:
    """依次尝试五类单编辑，去除仍成功及序列重复的候选。"""

    actions = _validate_request(grid, base, max_actions)
    # 返回副本，调用方调整列表不会污染缓存和后续生成的确定性。
    return list(_single_pool(grid, actions, max_actions))


def find_double_corruption(
    grid: Grid,
    base: Sequence[str],
    reason: ErrorReason | str,
    rng: random.Random,
    max_attempts: int = 500,
    max_actions: int = 72,
    avoid: Sequence[str] | None = None,
) -> Corruption | None:
    """采样两次独立原索引编辑，保留单独与联合均无效的反事实证据。"""

    if type(max_attempts) is not int or max_attempts < 0:
        raise ValueError("max_attempts 必须为非负整数")
    target_reason = ErrorReason(reason)
    actions = _validate_request(grid, base, max_actions)
    if avoid is not None and (
        not isinstance(avoid, Sequence)
        or isinstance(avoid, (str, bytes, bytearray))
        or any(not isinstance(token, str) for token in avoid)
    ):
        raise TypeError("avoid 必须为候选动作 token 序列")
    avoided = None if avoid is None else tuple(avoid)
    if max_attempts == 0:
        return None

    pool = tuple(item for item in _single_pool(grid, actions, max_actions)
                 if item.edits[0].kind != "truncate")
    if len({item.edits[0].index for item in pool}) < 2:
        return None
    seen_pairs: set[tuple[Edit, Edit]] = set()
    for _ in range(max_attempts):
        selected = sorted(rng.sample(pool, 2), key=lambda item: item.edits[0].index)
        first, second = (item.edits[0] for item in selected)
        if first.index == second.index:
            continue
        pair = first, second
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        candidate = apply_edits(actions, pair)
        singles = tuple(item.actions for item in selected)
        if (len(candidate) > max_actions or candidate == actions
                or candidate in singles or candidate == avoided):
            continue
        result = execute(grid, candidate)
        if not result.success and result.error_reason == target_reason:
            return Corruption(candidate, pair, singles)
    return None
