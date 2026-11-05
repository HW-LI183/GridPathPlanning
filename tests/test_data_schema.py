"""验证环境标识、布局隔离、网格回读和任务输入的信息边界。"""

import json
import re

import pytest

from gridpath.data_schema import (
    canonical_hash,
    environment_id,
    grid_from_record,
    grid_to_dict,
    layout_id,
    layout_split,
)
from gridpath.prompts import (
    describe_grid,
    render_generation,
    render_repair,
)
from gridpath.types import Grid


@pytest.fixture
def grid():
    return Grid(3, 4, [(2, 2), (0, 2), (1, 1)], (0, 0), (2, 3))


def test_canonical_hash_is_independent_of_dictionary_insertion_order():
    first = canonical_hash({"z": [3, 2], "a": {"y": 1, "x": 2}}, "sample")
    second = canonical_hash({"a": {"x": 2, "y": 1}, "z": [3, 2]}, "sample")

    assert first == second
    assert re.fullmatch(r"sample_[0-9a-f]{24}", first)
    assert first != canonical_hash({"z": [2, 3], "a": {"x": 2, "y": 1}}, "sample")


def test_layout_hash_sorts_obstacles_and_excludes_start_and_goal(grid):
    reordered = Grid(3, 4, [(1, 1), (0, 2), (2, 2)], (0, 0), (2, 3))
    other_task = Grid(3, 4, grid.obstacles, (1, 0), (1, 3))
    other_size = Grid(4, 4, grid.obstacles, grid.start, grid.goal)

    assert layout_id(grid) == layout_id(reordered) == layout_id(other_task)
    assert layout_id(grid) != layout_id(other_size)
    assert environment_id(grid) == environment_id(reordered)
    assert environment_id(grid) != environment_id(other_task)


def test_layout_owner_is_stable_before_path_generation(grid):
    identifier = layout_id(grid)
    owner = layout_split(identifier, 4051)

    assert owner in {"train", "dev", "test"}
    assert layout_split(identifier, 4051) == owner
    assert layout_split(identifier, 4051, {"test": 10, "train": 80, "dev": 10}) == owner
    other_task = Grid(3, 4, grid.obstacles, (2, 0), (0, 3))
    assert layout_split(layout_id(other_task), 4051) == owner
    assert layout_split(identifier, 4051, {"train": 0, "dev": 1, "test": 0}) == "dev"


@pytest.mark.parametrize("ratios", [{}, {"train": 0}, {"train": -1}, {"train": 1.5}, {"train": True}])
def test_layout_split_rejects_invalid_weights(grid, ratios):
    with pytest.raises(ValueError):
        layout_split(layout_id(grid), 4051, ratios)


def test_grid_record_round_trips_through_json_without_reference_fields(grid):
    record = grid_to_dict(grid)

    assert record == {
        "grid_size": [3, 4],
        "obstacles": [[0, 2], [1, 1], [2, 2]],
        "start": [0, 0],
        "goal": [2, 3],
    }
    assert grid_from_record(json.loads(json.dumps(record))) == grid
    assert grid_from_record({"grid": record, "reference_plan": ["ORACLE_ACTION"]}) == grid


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"grid": "not a grid"},
        {"grid_size": [3], "obstacles": [], "start": [0, 0], "goal": [1, 1]},
        {"grid_size": [3, 3], "obstacles": [1], "start": [0, 0], "goal": [1, 1]},
    ],
)
def test_grid_record_rejects_missing_or_malformed_fields(record):
    with pytest.raises(ValueError):
        grid_from_record(record)


def test_grid_description_has_directions_coordinates_and_full_sequence_semantics(grid):
    description = describe_grid(grid)

    assert "Zero-based ( row , col )" in description
    assert "up row minus 1; down row plus 1" in description
    assert "Obstacles ( 0 , 2 ) ; ( 1 , 1 ) ; ( 2 , 2 )." in description
    assert "Start ( 0 , 0 ). Goal ( 2 , 3 )." in description
    assert "Execute all actions; end at the goal." in description


def test_task_inputs_never_consume_reference_actions_or_distances(grid):
    record = {
        **grid_to_dict(grid),
        "reference_plan": ["ORACLE_ACTION"],
        "shortest_distance": 9857,
        "reference_trace": "ORACLE_TRACE",
    }
    restored = grid_from_record(record)
    candidate = ("down", "jump")
    inputs = [
        render_generation(restored),
        render_repair(restored, candidate, "invalid | step=2 | reason=invalid_action_token"),
    ]

    for prefix, text in zip(("GENERATE", "REPAIR"), inputs, strict=True):
        assert text.startswith(f"{prefix}\n")
        assert "ORACLE_ACTION" not in text
        assert "ORACLE_TRACE" not in text
        assert "9857" not in text
        assert "shortest_distance" not in text
        assert "reference_plan" not in text
    assert "Candidate:" not in inputs[0]
    assert "Candidate: down jump\n" in inputs[1]


def test_empty_candidates_remain_empty_and_valid_feedback_is_supported(grid):
    repair = render_repair(grid, (), " VALID ")

    assert "Candidate: \nFeedback: valid\n" in repair
    assert "Candidate: invalid_action_token" not in repair


@pytest.mark.parametrize(
    "feedback",
    [
        "invalid | step=1 | reason=out_of_bounds | shortest_distance=2",
        "invalid | step=1 | reason=out_of_bounds\nReference plan: right right",
        "valid | trace=(0,0)",
    ],
)
def test_repair_feedback_rejects_oracle_fields_and_explanatory_text(grid, feedback):
    with pytest.raises(ValueError):
        render_repair(grid, ("up",), feedback)
