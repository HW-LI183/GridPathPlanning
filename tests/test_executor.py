"""覆盖失败位置、终点后缀和结构化轨迹的语义边界。"""

import json

import pytest

from gridpath.executor import execute, normalize_actions
from gridpath.types import ErrorReason, Grid


@pytest.fixture
def grid():
    return Grid.from_world([[2, 3, 0], [0, 1, 0], [0, 0, 0]])


@pytest.mark.parametrize("plan,success,feasible,step,reason,position,executed", [
    ("right", True, True, None, None, (0, 1), 1),
    ("right left", False, True, 3, "goal_not_reached", (0, 0), 2),
    ("right left right", True, True, None, None, (0, 1), 3),
    ("right down", False, False, 2, "obstacle_collision", (0, 1), 1),
    ("left", False, False, 1, "out_of_bounds", (0, 0), 0),
    ("right up", False, False, 2, "out_of_bounds", (0, 1), 1),
    ("down right", False, False, 2, "obstacle_collision", (1, 0), 1),
    ("down", False, True, 2, "goal_not_reached", (1, 0), 1),
    ("", False, True, 1, "goal_not_reached", (0, 0), 0),
    (" \n\t ", False, True, 1, "goal_not_reached", (0, 0), 0),
    ("jump right", False, False, 1, "invalid_action_token", (0, 0), 0),
    ("right jump", False, False, 2, "invalid_action_token", (0, 1), 1),
    ("right,", False, False, 1, "invalid_action_token", (0, 0), 0),
    ("rightleft", False, False, 1, "invalid_action_token", (0, 0), 0),
    (" RIGHT\n\t ", True, True, None, None, (0, 1), 1),
])
def test_execution_contract(grid, plan, success, feasible, step, reason, position, executed):
    result = execute(grid, plan)
    assert (result.success, result.feasible, result.first_error_step, result.error_reason,
            result.final_position, result.executed_steps) == (success, feasible, step, reason, position, executed)
    expected = "valid" if success else f"invalid | step={step} | reason={reason}"
    assert result.verification_label == expected


@pytest.mark.parametrize("start,plan,step", [((0, 0), "up", 1), ((0, 0), "left", 1),
                                         ((2, 2), "down", 1), ((2, 2), "right", 1)])
def test_all_four_boundaries(start, plan, step):
    result = execute(Grid(3, 3, frozenset(), start, (1, 1)), plan)
    assert result.error_reason == ErrorReason.OUT_OF_BOUNDS
    assert result.first_error_step == step
    assert result.final_position == start


def test_failure_trace_keeps_last_legal_position_and_unexecuted_suffix(grid):
    result = execute(grid, "down right jump up")
    assert result.actions == ("down", "right", "jump", "up")
    assert len(result.full_trace) == 2
    failed = result.full_trace[-1]
    assert (failed.position_before, failed.attempted_position, failed.position_after) == ((1, 0), (1, 1), (1, 0))
    assert failed.error_reason == ErrorReason.OBSTACLE_COLLISION
    assert result.first_error_step == 2  # 不继续把未执行的 jump 误报为首错。


def test_invalid_token_has_no_attempted_move(grid):
    failure = execute(grid, "jump").full_trace[0]
    assert failure.attempted_position is None
    assert failure.position_before == failure.position_after == grid.start


def test_goal_not_reached_adds_explicit_end_event(grid):
    result = execute(grid, "down")
    end = result.full_trace[-1]
    assert (end.step, end.action, end.attempted_position) == (2, None, None)
    assert end.error_reason == ErrorReason.GOAL_NOT_REACHED
    assert len(result.full_trace) == result.executed_steps + 1


def test_start_equals_goal_and_suffix_still_matters():
    grid = Grid(1, 2, frozenset(), (0, 0), (0, 0))
    empty = execute(grid, [])
    assert empty.success and empty.executed_steps == 0 and empty.full_trace == ()
    assert execute(grid, "right").error_reason == ErrorReason.GOAL_NOT_REACHED
    assert execute(grid, "right left").success


def test_normalization_accepts_sequences_and_rejects_nontext():
    assert normalize_actions(["UP", " down\n", " ", "left"]) == ("up", "down", "left")
    with pytest.raises(TypeError):
        normalize_actions(["up", 1])


def test_result_can_be_saved_as_json_without_custom_encoder(grid):
    saved = json.loads(json.dumps(execute(grid, "jump").to_dict()))
    assert saved["error_reason"] == "invalid_action_token"
    assert saved["full_trace"][0]["position_after"] == [0, 0]
