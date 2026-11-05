"""网格输入校验和官方矩阵表示的边界检查。"""

import pytest

from gridpath.types import Grid


def test_normalized_immutable_grid_and_world_round_trip():
    grid = Grid(2, 3, [(1, 1), (1, 1)], [0, 0], [1, 2])
    assert grid.obstacles == frozenset({(1, 1)})
    assert grid.start == (0, 0)
    assert Grid.from_world(grid.to_world()) == grid


@pytest.mark.parametrize("rows,cols,obstacles,start,goal", [
    (0, 3, [], (0, 0), (0, 1)),
    (True, 3, [], (0, 0), (0, 1)),
    (3, 3, [], (-1, 0), (0, 1)),
    (3, 3, [(0, 0)], (0, 0), (0, 1)),
    (3, 3, [(0, 1)], (0, 0), (0, 1)),
    (3, 3, [(3, 1)], (0, 0), (0, 1)),
    (3, 3, [], (0.0, 0), (0, 1)),
])
def test_reject_malformed_grid(rows, cols, obstacles, start, goal):
    with pytest.raises(ValueError):
        Grid(rows, cols, obstacles, start, goal)


@pytest.mark.parametrize("world", [[], [[]], [[2, 3], [0]], [[2, 9, 3]], [[2, 3, 3]], [[0, 3]], [[2, True, 3]]])
def test_reject_malformed_world(world):
    with pytest.raises(ValueError):
        Grid.from_world(world)


def test_single_cell_start_goal_uses_explicit_start():
    grid = Grid(1, 1, frozenset(), (0, 0), (0, 0))
    assert grid.to_world() == [[3]]
    assert Grid.from_world(grid.to_world(), start=grid.start) == grid


@pytest.mark.parametrize("world,kwargs", [
    ([[2, 0, 3]], {"start": (0, 1)}),
    ([[2, 0, 3]], {"goal": (0, 1)}),
    ([[2, 2, 3]], {"start": (0, 0)}),
    ([[2, 3, 3]], {"goal": (0, 1)}),
])
def test_explicit_coordinates_cannot_hide_conflicting_or_duplicate_markers(world, kwargs):
    with pytest.raises(ValueError):
        Grid.from_world(world, **kwargs)


def test_explicit_coordinates_can_fill_missing_markers_or_confirm_matching_ones():
    expected = Grid(1, 3, frozenset(), (0, 0), (0, 2))
    assert Grid.from_world([[0, 0, 0]], start=[0, 0], goal=[0, 2]) == expected
    assert Grid.from_world([[2, 0, 3]], start=[0, 0], goal=[0, 2]) == expected
