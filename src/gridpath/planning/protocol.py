"""推理表示与预测聚合；不调用执行器、不读取标签、不检查环境合法性。"""
from ..types import ACTION_DELTAS

TAGS = ('continue', 'obstacle_collision', 'out_of_bounds',
        'invalid_action_token', 'valid_end', 'goal_not_reached')
ACTION_MARKER = '<extra_id_0>'
END_MARKER = '<extra_id_1>'


def actions_of(candidate):
    if not isinstance(candidate, str):
        candidate = ' '.join(candidate)
    return candidate.lower().split()


def virtual_states(start, candidate):
    """只累加位移：穿过障碍、超出边界仍继续；未知词占位移0，不作合法性判断。"""
    position = tuple(start)
    states = [position]
    for action in actions_of(candidate):
        dr, dc = ACTION_DELTAS.get(action, (0, 0))
        position = (position[0] + dr, position[1] + dc)
        states.append(position)
    return states


def render(grid, candidate):
    actions = actions_of(candidate)
    states = virtual_states(grid.start, actions)
    obstacles = '; '.join(f'row {r} col {c}' for r, c in sorted(grid.obstacles)) or 'none'
    lines = [f'Verify all actions. Grid rows {grid.rows} cols {grid.cols}. '
             f'Start row {grid.start[0]} col {grid.start[1]}. '
             f'Goal row {grid.goal[0]} col {grid.goal[1]}. Obstacles: {obstacles}.',
             'Coordinates below are unchecked displacement sums, not legal execution.']
    for i, action in enumerate(actions):
        before, after = states[i:i+2]
        offsets = '; '.join(f'dr {r-after[0]} dc {c-after[1]}'
                            for r,c in sorted(grid.obstacles)) or 'none'
        lines.append(f'{ACTION_MARKER} step {i+1} from row {before[0]} col {before[1]} '
                     f'action {action} to row {after[0]} col {after[1]}. '
                     f'Obstacle minus proposed offsets: {offsets}.')
    lines.append(f'{END_MARKER} END at row {states[-1][0]} col {states[-1][1]}; '
                 f'goal row {grid.goal[0]} col {grid.goal[1]}; '
                 f'goal minus end dr {grid.goal[0]-states[-1][0]} dc {grid.goal[1]-states[-1][1]}.')
    return '\n'.join(lines)


def feature(tokenizer, grid, candidate, limit, labels=None):
    actions = actions_of(candidate)
    text = render(grid, actions)
    ids = tokenizer(text, truncation=False)['input_ids']
    if len(ids) > limit:
        raise ValueError(f'验证输入{len(ids)} token超过{limit}，拒绝截断')
    a, e = (tokenizer.convert_tokens_to_ids(x) for x in (ACTION_MARKER, END_MARKER))
    positions = [i for i, x in enumerate(ids) if x in (a, e)]
    if len(positions) != len(actions)+1 or ids[positions[-1]] != e:
        raise ValueError('候选含保留标记或标记位置异常')
    result = {'input_ids': ids, 'positions': positions,
              'kinds': [0]*len(actions)+[1]}
    if labels is not None:
        if len(labels) != len(positions):
            raise ValueError('路径内监督数量不符')
        result['labels'] = labels
    return result


def aggregate(tags, action_count):
    """位置来自第一个模型预测错误，类型也来自模型；规则不读取网格。"""
    if len(tags) != action_count+1:
        raise ValueError('预测标记数不符')
    for i, tag in enumerate(tags[:-1]):
        if tag not in range(4):
            raise ValueError('动作位置类别不合法')
        if tag != 0:
            return f'invalid | step={i+1} | reason={TAGS[tag]}'
    if tags[-1] == 4:
        return 'valid'
    if tags[-1] == 5:
        return f'invalid | step={action_count+1} | reason=goal_not_reached'
    raise ValueError('END类别不合法')
