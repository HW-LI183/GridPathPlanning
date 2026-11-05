"""检查原索引编辑、候选去重和双错误的反事实审计条件。"""

from dataclasses import FrozenInstanceError
import random

import pytest

from gridpath.corruptions import (
    Corruption,
    Edit,
    apply_edits,
    edit_from_dict,
    edit_to_dict,
    enumerate_single_corruptions,
    find_double_corruption,
)
from gridpath.executor import execute
from gridpath.types import ErrorReason, Grid


@pytest.fixture
def example():
    return Grid(4, 4, frozenset({(0, 1)}), (0, 0), (2, 2)), ("down", "down", "right", "right")


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (Edit("substitute", 1, "left"), ("right", "left", "right")),
        (Edit("delete", 1), ("right", "right")),
        (Edit("insert", 1, "up"), ("right", "up", "down", "right")),
        (Edit("insert", 3, "up"), ("right", "down", "right", "up")),
        (Edit("truncate", 1), ("right",)),
        (Edit("truncate", 0), ()),
        (Edit("invalid_token", 1, "jump"), ("right", "jump", "right")),
    ],
)
def test_each_edit_has_explicit_original_index_semantics(edit, expected):
    assert apply_edits(("right", "down", "right"), (edit,)) == expected


def test_delete_and_substitute_do_not_shift_original_indices():
    base = ("right", "down", "right", "down")
    edits = (Edit("delete", 0), Edit("substitute", 2, "up"))

    assert apply_edits(base, edits) == ("down", "up", "down")
    assert apply_edits(base, tuple(reversed(edits))) == ("down", "up", "down")


def test_insert_and_delete_preserve_both_distinct_edits():
    edits = (Edit("insert", 1, "left"), Edit("delete", 3))

    assert apply_edits(("right", "down", "right", "down"), edits) == (
        "right", "left", "down", "right"
    )


@pytest.mark.parametrize(
    "edits",
    [
        (Edit("delete", 0), Edit("insert", 0, "down")),
        (Edit("substitute", 0, "up"), Edit("invalid_token", 0, "jump")),
        (Edit("truncate", 1), Edit("substitute", 2, "down")),
        (Edit("delete", 0), Edit("truncate", 2)),
        (Edit("delete", 3),),
        (Edit("insert", 4, "up"),),
        (Edit("truncate", 3),),
        (Edit("substitute", 0, "right"),),
    ],
)
def test_overlap_truncation_out_of_range_and_noop_edits_are_rejected(edits):
    with pytest.raises(ValueError):
        apply_edits(("right", "down", "right"), edits)


@pytest.mark.parametrize(
    "fields",
    [
        ("unknown", 0, None),
        ("delete", -1, None),
        ("delete", True, None),
        ("delete", 0, "up"),
        ("insert", 0, "jump"),
        ("substitute", 0, None),
        ("substitute", 0, "UP"),
        ("invalid_token", 0, "up"),
        ("invalid_token", 0, "jump down"),
        ("invalid_token", 0, ""),
    ],
)
def test_malformed_edit_is_rejected(fields):
    with pytest.raises(ValueError):
        Edit(*fields)


def test_edit_logs_round_trip_and_are_immutable():
    edits = (Edit("delete", 1), Edit("invalid_token", 2, "jump"))
    for edit in edits:
        assert edit_from_dict(edit_to_dict(edit)) == edit
    assert edit_from_dict({"kind": "delete", "index": 1}) == edits[0]
    with pytest.raises(ValueError):
        edit_from_dict({"kind": "delete", "index": 1, "ignored": "bad"})
    with pytest.raises(FrozenInstanceError):
        edits[0].index = 2


def test_single_pool_is_deterministic_unique_invalid_and_auditable(example):
    grid, base = example

    pool = enumerate_single_corruptions(grid, base)

    assert {item.edits[0].kind for item in pool} == {
        "substitute", "delete", "insert", "truncate", "invalid_token"
    }
    assert len(pool) <= 10 * len(base) + 4
    assert len({item.actions for item in pool}) == len(pool)
    for item in pool:
        assert len(item.edits) == 1
        assert item.actions != base
        assert item.actions == apply_edits(base, item.edits)
        assert item.single_actions == (item.actions,)
        assert not execute(grid, item.actions).success
    expected = tuple(pool)
    pool.clear()
    assert tuple(enumerate_single_corruptions(grid, base)) == expected


def test_truncation_that_remains_valid_is_filtered_out():
    grid = Grid(2, 2, frozenset(), (0, 0), (0, 1))
    base = ("right", "down", "up")

    pool = enumerate_single_corruptions(grid, base)

    assert ("right",) not in {item.actions for item in pool}


def test_max_actions_bounds_both_single_and_double_candidates(example):
    grid, base = example

    pool = enumerate_single_corruptions(grid, base, max_actions=len(base))
    double = find_double_corruption(
        grid, base, ErrorReason.INVALID_ACTION_TOKEN, random.Random(3), max_actions=len(base)
    )

    assert pool
    assert all(len(item.actions) <= len(base) for item in pool)
    assert all(item.edits[0].kind != "insert" for item in pool)
    assert double is not None
    assert len(double.actions) <= len(base)


@pytest.mark.parametrize("reason", list(ErrorReason))
def test_double_corruption_preserves_two_independent_invalid_edits(example, reason):
    grid, base = example

    result = find_double_corruption(grid, base, reason, random.Random(4051))

    assert isinstance(result, Corruption)
    assert len(result.edits) == 2
    assert result.edits[0].index != result.edits[1].index
    assert all(edit.kind != "truncate" for edit in result.edits)
    assert result.actions == apply_edits(base, result.edits)
    assert result.actions != base
    assert result.single_actions == tuple(apply_edits(base, (edit,)) for edit in result.edits)
    assert result.actions not in result.single_actions
    assert all(not execute(grid, actions).success for actions in result.single_actions)
    execution = execute(grid, result.actions)
    assert not execution.success
    assert execution.error_reason == reason
    assert find_double_corruption(grid, base, reason.value, random.Random(4051)) == result


class FixedEditsRandom(random.Random):
    """固定选中已存在的编辑对，测试拒绝逻辑不依赖采样概率。"""

    def __init__(self, edits):
        super().__init__(0)
        self.edits = edits

    def sample(self, population, k, *, counts=None):
        assert k == 2
        by_edit = {item.edits[0]: item for item in population}
        return [by_edit[edit] for edit in self.edits]


def test_jointly_valid_two_edits_are_rejected():
    grid = Grid(4, 4, frozenset(), (0, 0), (2, 2))
    base = ("right", "right", "down", "down")
    edits = (Edit("substitute", 0, "down"), Edit("substitute", 2, "right"))
    assert all(not execute(grid, apply_edits(base, (edit,))).success for edit in edits)
    assert execute(grid, apply_edits(base, edits)).success

    result = find_double_corruption(
        grid, base, "goal_not_reached", FixedEditsRandom(edits), max_attempts=1
    )

    assert result is None


def test_avoid_excludes_the_matching_id_candidate(example):
    grid, base = example
    result = find_double_corruption(grid, base, "invalid_action_token", random.Random(1))
    assert result is not None

    avoided = find_double_corruption(
        grid, base, "invalid_action_token", FixedEditsRandom(result.edits),
        max_attempts=1, avoid=result.actions
    )

    assert avoided is None


def test_wrong_first_error_reason_is_rejected(example):
    grid, base = example
    result = find_double_corruption(grid, base, "invalid_action_token", random.Random(1))
    assert result is not None

    wrong_reason = find_double_corruption(
        grid, base, "goal_not_reached", FixedEditsRandom(result.edits), max_attempts=1
    )

    assert wrong_reason is None


def test_empty_successful_base_has_bounded_single_edits_but_no_double():
    grid = Grid(1, 1, frozenset(), (0, 0), (0, 0))

    pool = enumerate_single_corruptions(grid, ())

    assert len(pool) == 4
    assert all(item.edits[0].kind == "insert" for item in pool)
    assert find_double_corruption(grid, (), "out_of_bounds", random.Random(0)) is None
    assert enumerate_single_corruptions(grid, (), max_actions=0) == []


def test_invalid_base_and_bad_budgets_are_rejected(example):
    grid, base = example
    with pytest.raises(ValueError):
        enumerate_single_corruptions(grid, ("up",))
    with pytest.raises(ValueError):
        enumerate_single_corruptions(grid, ("DOWN", "down", "right", "right"))
    with pytest.raises(TypeError):
        apply_edits("down down right right", ())
    with pytest.raises(ValueError):
        enumerate_single_corruptions(grid, base, max_actions=3)
    with pytest.raises(ValueError):
        find_double_corruption(grid, base, "unknown", random.Random(0))
    with pytest.raises(ValueError):
        find_double_corruption(grid, base, "out_of_bounds", random.Random(0), max_attempts=-1)
    assert find_double_corruption(
        grid, base, "out_of_bounds", random.Random(0), max_attempts=0
    ) is None
