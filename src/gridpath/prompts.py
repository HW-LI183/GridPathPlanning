"""v1固定boundary表示；不读取参考动作、标签或最短距离。"""
from .types import Grid
from .executor import normalize_actions
from .metrics import parse_verification_label


def describe_grid(grid: Grid):
    coord = lambda p: f"( {p[0]} , {p[1]} )"
    obstacles = " ; ".join(coord(p) for p in sorted(grid.obstacles)) or "none"
    return (f"Grid {grid.rows} by {grid.cols}. Zero-based ( row , col ). "
        "Rows increase down; columns increase right. "
        f"Start {coord(grid.start)}. Goal {coord(grid.goal)}. Obstacles {obstacles}.\n"
        "Actions: up row minus 1; down row plus 1; left col minus 1; right col plus 1. "
        "Stay inside the grid; avoid obstacles. Execute all actions; end at the goal.\n")


def render_generation(grid):
    return "GENERATE\n" + describe_grid(grid) + "Return only a shortest path as space-separated actions."


def render_repair(grid, candidate, feedback):
    label=parse_verification_label(feedback)
    if label is None:
        raise ValueError("修复输入必须有规范执行器反馈")
    feedback="valid" if label.valid else f"invalid | step={label.step} | reason={label.reason.value}"
    return ("REPAIR\n" + describe_grid(grid) + "Candidate: " + " ".join(normalize_actions(candidate))
        + f"\nFeedback: {feedback}\nIf valid, copy candidate. Otherwise return the complete corrected path, actions only.")
