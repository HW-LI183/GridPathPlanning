"""保留官方完整序列终点语义，显式补充严格词法和首次错误轨迹。"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .types import ACTION_DELTAS, Coord, ErrorReason, Grid


@dataclass(frozen=True)
class TraceStep:
    step: int
    action: str | None
    position_before: Coord
    attempted_position: Coord | None
    position_after: Coord
    error_reason: ErrorReason | None = None


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    feasible: bool
    first_error_step: int | None
    error_reason: ErrorReason | None
    final_position: Coord
    executed_steps: int
    full_trace: tuple[TraceStep, ...]
    actions: tuple[str, ...]

    @property
    def verification_label(self) -> str:
        if self.success:
            return "valid"
        return f"invalid | step={self.first_error_step} | reason={self.error_reason.value}"

    def to_dict(self) -> dict:
        return {**asdict(self), "verification_label": self.verification_label}


def normalize_actions(plan: str | Sequence[str]) -> tuple[str, ...]:
    """只归一大小写和空白；不删除标点、不补全或拆分粘连方向词。"""
    if isinstance(plan, str):
        return tuple(token.lower() for token in plan.split())
    if (not isinstance(plan, Sequence) or isinstance(plan, (bytes, bytearray))
            or any(not isinstance(token, str) for token in plan)):
        raise TypeError("plan 必须为文本或动作文本序列")
    return normalize_actions(" ".join(plan))


def execute(grid: Grid, plan: str | Sequence[str]) -> ExecutionResult:
    """遇到首个非法动作即停止；通过目标时仍执行后缀，终点才决定成功。"""
    actions = normalize_actions(plan)
    position = grid.start
    trace: list[TraceStep] = []
    executed_steps = 0
    for step, action in enumerate(actions, start=1):
        attempted = None
        if action not in ACTION_DELTAS:
            reason = ErrorReason.INVALID_ACTION_TOKEN
        else:
            row_delta, col_delta = ACTION_DELTAS[action]
            attempted = position[0] + row_delta, position[1] + col_delta
            if not grid.in_bounds(attempted):
                reason = ErrorReason.OUT_OF_BOUNDS
            elif attempted in grid.obstacles:
                reason = ErrorReason.OBSTACLE_COLLISION
            else:
                reason = None
        if reason is not None:
            trace.append(TraceStep(step, action, position, attempted, position, reason))
            return ExecutionResult(False, False, step, reason, position,
                                   executed_steps, tuple(trace), actions)
        before = position
        position = attempted
        executed_steps += 1
        trace.append(TraceStep(step, action, before, position, position))

    if position == grid.goal:
        return ExecutionResult(True, True, None, None, position, executed_steps, tuple(trace), actions)

    step = len(actions) + 1
    reason = ErrorReason.GOAL_NOT_REACHED
    trace.append(TraceStep(step, None, position, None, position, reason))
    return ExecutionResult(False, True, step, reason, position, executed_steps, tuple(trace), actions)
