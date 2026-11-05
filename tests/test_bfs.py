"""验证独立 BFS 的最短距离、稳定路径和边界情形。"""

from dataclasses import FrozenInstanceError
import random

import pytest

from gridpath.bfs import ShortestPath, shortest_path
from gridpath.types import Grid


def make_grid(
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: tuple[tuple[int, int], ...] = (),
) -> Grid:
    return Grid(rows, cols, frozenset(obstacles), start, goal)


def assert_legal_path(grid: Grid, result: ShortestPath) -> None:
    """独立回放动作，检查位置、边界和障碍约束。"""

    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    assert result.reachable
    assert result.distance == len(result.actions)
    assert len(result.positions) == result.distance + 1
    assert result.positions[0] == grid.start
    assert result.positions[-1] == grid.goal
    current = grid.start
    for action, expected_position in zip(result.actions, result.positions[1:]):
        delta_row, delta_col = deltas[action]
        current = (current[0] + delta_row, current[1] + delta_col)
        assert current == expected_position
        assert 0 <= current[0] < grid.rows
        assert 0 <= current[1] < grid.cols
        assert current not in grid.obstacles


def reference_distance(grid: Grid) -> int | None:
    """通过反复松弛全部可通行边计算距离，独立于队列式 BFS。"""

    cells = [
        (row, col)
        for row in range(grid.rows)
        for col in range(grid.cols)
        if (row, col) not in grid.obstacles
    ]
    distances = {grid.start: 0}
    for _ in range(len(cells)):
        changed = False
        for row, col in cells:
            candidates = [
                distances[neighbor] + 1
                for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                if neighbor in distances
            ]
            if candidates and min(candidates) < distances.get((row, col), float("inf")):
                distances[(row, col)] = min(candidates)
                changed = True
        if not changed:
            break
    return distances.get(grid.goal)


def test_shortest_path_uses_fixed_tie_break() -> None:
    grid = make_grid(3, 3, (1, 1), (0, 2))

    result = shortest_path(grid)

    assert result.actions == ("up", "right")
    assert result.positions == ((1, 1), (0, 1), (0, 2))
    assert result.distance == 2
    assert shortest_path(grid) == result
    assert_legal_path(grid, result)


def test_obstacles_require_a_longer_detour() -> None:
    grid = make_grid(3, 5, (0, 0), (0, 4), ((0, 2), (1, 2)))

    result = shortest_path(grid)

    assert result.distance == 8
    assert result.positions[-1] == (0, 4)
    assert_legal_path(grid, result)


def test_disconnected_goal_returns_no_path() -> None:
    grid = make_grid(3, 4, (0, 0), (2, 3), ((1, 0), (1, 1), (1, 2), (1, 3)))

    result = shortest_path(grid)

    assert result == ShortestPath(False, None, (), ())


@pytest.mark.parametrize(
    ("rows", "cols", "start", "goal", "distance"),
    [
        (1, 5, (0, 0), (0, 4), 4),
        (5, 1, (4, 0), (0, 0), 4),
        (2, 6, (1, 0), (0, 5), 6),
        (6, 2, (0, 1), (5, 0), 6),
    ],
)
def test_rectangular_and_single_axis_grids(rows, cols, start, goal, distance) -> None:
    grid = make_grid(rows, cols, start, goal)

    result = shortest_path(grid)

    assert result.distance == distance
    assert_legal_path(grid, result)


def test_start_equal_to_goal_needs_no_actions() -> None:
    grid = make_grid(1, 1, (0, 0), (0, 0))

    result = shortest_path(grid)

    assert result == ShortestPath(True, 0, (), ((0, 0),))
    assert_legal_path(grid, result)


def test_result_is_immutable() -> None:
    result = shortest_path(make_grid(1, 2, (0, 0), (0, 1)))

    with pytest.raises(FrozenInstanceError):
        result.distance = 100


def test_random_small_grids_match_independent_distance_reference() -> None:
    random_source = random.Random(4051)
    for _ in range(80):
        rows = random_source.randint(1, 5)
        cols = random_source.randint(1, 6)
        cells = [(row, col) for row in range(rows) for col in range(cols)]
        start = random_source.choice(cells)
        goal = random_source.choice(cells)
        obstacles = tuple(
            cell for cell in cells if cell not in (start, goal) and random_source.random() < 0.35
        )
        grid = make_grid(rows, cols, start, goal, obstacles)

        result = shortest_path(grid)
        distance = reference_distance(grid)

        assert result.distance == distance
        assert result.reachable == (distance is not None)
        if result.reachable:
            assert_legal_path(grid, result)
        else:
            assert result.actions == ()
            assert result.positions == ()
