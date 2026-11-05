"""仅用于数据审计和评估的四邻域最短路，不参与模型推理。"""

from collections import deque
from dataclasses import dataclass

from .types import ACTION_DELTAS, ACTION_ORDER, Coord, Grid


@dataclass(frozen=True)
class ShortestPath:
    """最短路结果；距离采用动作数，位置序列包含起点和终点。"""

    reachable: bool
    distance: int | None
    actions: tuple[str, ...]
    positions: tuple[Coord, ...]


def shortest_path(grid: Grid) -> ShortestPath:
    """按固定动作顺序搜索最短路，不可达时返回空路径。"""

    # 首次发现位置时记录前驱，避免同一位置重复入队。
    predecessors: dict[Coord, tuple[Coord, str] | None] = {grid.start: None}
    frontier = deque([grid.start])

    while frontier:
        current = frontier.popleft()
        if current == grid.goal:
            actions: list[str] = []
            positions = [current]
            while predecessors[current] is not None:
                previous, action = predecessors[current]
                actions.append(action)
                positions.append(previous)
                current = previous
            actions.reverse()
            positions.reverse()
            return ShortestPath(
                reachable=True,
                distance=len(actions),
                actions=tuple(actions),
                positions=tuple(positions),
            )

        for action in ACTION_ORDER:
            delta_row, delta_col = ACTION_DELTAS[action]
            neighbor = (current[0] + delta_row, current[1] + delta_col)
            if (
                grid.in_bounds(neighbor)
                and neighbor not in grid.obstacles
                and neighbor not in predecessors
            ):
                predecessors[neighbor] = (current, action)
                frontier.append(neighbor)

    return ShortestPath(False, None, (), ())
