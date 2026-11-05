"""网格、坐标与四类执行错误的统一定义。"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


Coord = tuple[int, int]
EXECUTION_PROTOCOL_VERSION = "gridpath.execution.v1"
ACTION_ORDER = ("up", "down", "left", "right")
ACTION_DELTAS = MappingProxyType({"up": (-1, 0), "down": (1, 0),
                                "left": (0, -1), "right": (0, 1)})


class ErrorReason(str, Enum):
    OBSTACLE_COLLISION = "obstacle_collision"
    OUT_OF_BOUNDS = "out_of_bounds"
    INVALID_ACTION_TOKEN = "invalid_action_token"
    GOAL_NOT_REACHED = "goal_not_reached"


def _coordinate(value, name: str) -> Coord:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != 2 or any(type(item) is not int for item in value)):
        raise ValueError(f"{name} 必须为两个整数构成的行列坐标")
    return value[0], value[1]


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    obstacles: frozenset[Coord]
    start: Coord
    goal: Coord

    def __post_init__(self):
        if type(self.rows) is not int or type(self.cols) is not int or min(self.rows, self.cols) <= 0:
            raise ValueError("网格行列数必须为正整数")
        object.__setattr__(self, "start", _coordinate(self.start, "start"))
        object.__setattr__(self, "goal", _coordinate(self.goal, "goal"))
        try:
            obstacles = frozenset(_coordinate(item, "obstacle") for item in self.obstacles)
        except TypeError as exc:
            raise ValueError("obstacles 必须为坐标集合") from exc
        object.__setattr__(self, "obstacles", obstacles)
        for coord in (self.start, self.goal, *self.obstacles):
            if not self.in_bounds(coord):
                raise ValueError(f"坐标超出网格范围：{coord}")
        if self.start in obstacles or self.goal in obstacles:
            raise ValueError("起点和目标不能位于障碍物上")

    def in_bounds(self, coord: Coord) -> bool:
        return 0 <= coord[0] < self.rows and 0 <= coord[1] < self.cols

    def to_world(self) -> list[list[int]]:
        """转为官方 0/1/2/3 矩阵；起终点重合时该格用 3，起点另传。"""
        world = [[0] * self.cols for _ in range(self.rows)]
        for row, col in self.obstacles:
            world[row][col] = 1
        world[self.start[0]][self.start[1]] = 2
        world[self.goal[0]][self.goal[1]] = 3
        return world

    @classmethod
    def from_world(cls, world: Sequence[Sequence[int]], *, start: Coord | None = None,
                   goal: Coord | None = None) -> "Grid":
        """读取官方矩阵；显式坐标可用于只有一个目标格的重合边界。"""
        if (not isinstance(world, Sequence) or isinstance(world, (str, bytes)) or not world
                or not isinstance(world[0], Sequence) or not world[0]):
            raise ValueError("world 必须为非空矩形二维序列")
        cols = len(world[0])
        starts, goals, obstacles = [], [], []
        for row, line in enumerate(world):
            if not isinstance(line, Sequence) or isinstance(line, (str, bytes)) or len(line) != cols:
                raise ValueError("world 必须为矩形二维序列")
            for col, cell in enumerate(line):
                if type(cell) is not int or cell not in (0, 1, 2, 3):
                    raise ValueError("world 单元只能使用整数 0/1/2/3")
                if cell == 1:
                    obstacles.append((row, col))
                elif cell == 2:
                    starts.append((row, col))
                elif cell == 3:
                    goals.append((row, col))
        if len(starts) > 1 or len(goals) > 1:
            raise ValueError("单目标 world 不允许重复起点或目标标记")
        if start is None:
            if len(starts) != 1:
                raise ValueError("world 必须包含唯一的起点，或显式传入 start")
            start = starts[0]
        else:
            start = _coordinate(start, "start")
            if starts and starts[0] != start:
                raise ValueError("显式 start 与 world 起点标记冲突")
        if goal is None:
            if len(goals) != 1:
                raise ValueError("world 必须包含唯一的目标，或显式传入 goal")
            goal = goals[0]
        else:
            goal = _coordinate(goal, "goal")
            if goals and goals[0] != goal:
                raise ValueError("显式 goal 与 world 目标标记冲突")
        return cls(len(world), cols, frozenset(obstacles), start, goal)
