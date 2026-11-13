"""普通网格描述＋候选；不包含位移轨迹、障碍偏移或逐步标签。"""
from ..prompts import describe_grid
from ..executor import normalize_actions


def render_verification(grid,candidate):
    return ('VERIFY\n'+describe_grid(grid)+'Candidate: '+' '.join(normalize_actions(candidate))+
        '\nJudge the entire candidate. Return valid if it ends at the goal without errors. '
        'Otherwise return invalid | step=N | reason=TYPE. Steps are one-based; '
        'if actions end before reaching the goal, step is action count plus one. '
        'TYPE is obstacle_collision, out_of_bounds, invalid_action_token, or goal_not_reached. '
        'Return only the label.')
