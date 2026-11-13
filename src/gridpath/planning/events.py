"""一次原子扰动≠一次token编辑；往返块两动作相互约束，OOD为两次独立事件。"""
from dataclasses import dataclass,asdict
from itertools import combinations
from ..types import ACTION_DELTAS,ACTION_ORDER,ErrorReason
from ..executor import execute
from ..corruptions import enumerate_single_corruptions
from ..training_config import digest
from .protocol import virtual_states

INVERSE={'up':'down','down':'up','left':'right','right':'left'}
KINDS=('substitute','delete','insert','truncate','invalid_token','return_block','insert_invalid_token')


@dataclass(frozen=True)
class Event:
    kind:str
    index:int
    tokens:tuple=()

    def __post_init__(self):
        if self.kind not in KINDS or type(self.index)!=int or self.index<0:raise ValueError('非法事件')
        if self.kind=='return_block' and (len(self.tokens)!=2 or self.tokens[0] not in INVERSE or self.tokens[1]!=INVERSE[self.tokens[0]]):
            raise ValueError('往返块必须是一对相反动作，不能将任意双编辑伪装为单错误')
        if self.kind in ('delete','truncate') and self.tokens:raise ValueError('删除/截断不得携带动作')
        if self.kind not in ('delete','truncate','return_block') and len(self.tokens)!=1:raise ValueError('事件动作数错误')
        if self.kind in ('substitute','insert') and self.tokens[0] not in ACTION_DELTAS:raise ValueError('应为方向动作')
        if self.kind in ('invalid_token','insert_invalid_token') and (self.tokens[0] in ACTION_DELTAS or self.tokens[0].split()!=[self.tokens[0]]):
            raise ValueError('无效词事件需要单个非法动作词')

    @property
    def primitive_token_edits(self):return 2 if self.kind=='return_block' else 1

    def to_dict(self):
        return {**asdict(self),'tokens':list(self.tokens),'primitive_token_edits':self.primitive_token_edits}


def event_from_dict(record):
    event=Event(record['kind'],record['index'],tuple(record['tokens']))
    if record!=event.to_dict():raise ValueError('事件字段与原始token编辑数不一致')
    return event


def apply_events(base,events):
    base=tuple(base);events=tuple(events)
    if any(a not in ACTION_DELTAS for a in base):raise ValueError('原路径只允许规范方向词')
    if len({e.index for e in events})!=len(events):raise ValueError('独立事件不能共用原路径索引')
    if any(e.kind=='truncate' for e in events):
        if len(events)!=1:raise ValueError('双错不允许用截断掩盖另一事件')
        if events[0].index>=len(base):raise ValueError('截断位置越界')
        return base[:events[0].index]
    by_index={e.index:e for e in events};out=[]
    for event in events:
        upper=len(base) if event.kind in ('insert','return_block','insert_invalid_token') else len(base)-1
        if event.index>upper:raise ValueError('事件索引超出原路径')
        if event.kind=='substitute' and event.tokens[0]==base[event.index]:raise ValueError('无效替换')
    for i in range(len(base)+1):
        event=by_index.get(i)
        if event and event.kind in ('insert','return_block','insert_invalid_token'):out.extend(event.tokens)
        if i==len(base):break
        if event is None or event.kind in ('insert','return_block','insert_invalid_token'):out.append(base[i])
        elif event.kind in ('substitute','invalid_token'):out.extend(event.tokens)
    return tuple(out)


def choices(grid,base,reason,kind=None):
    """只用于离线构造，不进入模型推理。每个候选保留完整反事实来源。"""
    reason=ErrorReason(reason);events=[]
    if kind=='return_block':
        events=[Event(kind,i,(a,INVERSE[a])) for i in range(len(base)+1) for a in ACTION_ORDER]
    elif kind=='insert_invalid_token':events=[Event(kind,i,('jump',)) for i in range(len(base)+1)]
    else:
        for c in enumerate_single_corruptions(grid,base):
            e=c.edits[0]
            if kind is not None and e.kind!=kind:continue
            token='jump' if e.token=='<invalid>' else e.token
            events.append(Event(e.kind,e.index,() if token is None else (token,)))
    return [e for e in events if (result:=execute(grid,apply_events(base,[e]))).error_reason==reason]


def safe_loop(grid,base,identity,at=None):
    events=[Event('return_block',i,(a,INVERSE[a])) for i in (range(len(base)+1) if at is None else [at]) for a in ACTION_ORDER]
    valid=[e for e in events if execute(grid,apply_events(base,[e])).success]
    if not valid:raise ValueError('没有合法的正例往返块')
    e=min(valid,key=lambda e:digest([42,'safe-loop',identity,e.to_dict()]))
    return apply_events(base,[e]),e


def goal_endpoint(grid,candidate):return virtual_states(grid.start,candidate)[-1]==grid.goal


def double_events(grid,base,reason,hard,identity):
    """逐个单独无效、不同原索引、联合无效且不等于任一单候选；无独立性假标签。"""
    pool=[]
    for target in ErrorReason:
        for kind in ('return_block','insert_invalid_token',None):
            pool.extend(choices(grid,base,target,kind))
    pool=sorted({e for e in pool if e.kind!='truncate'},key=lambda e:digest([42,identity,e.to_dict()]))
    singles={e:apply_events(base,[e]) for e in pool}
    for a,b in combinations(pool,2):
        if a.index==b.index:continue
        # 一个零净位移事件与一个非零事件不可能共同保持原终点。
        if hard and not (goal_endpoint(grid,singles[a]) and goal_endpoint(grid,singles[b])):continue
        combined=apply_events(base,[a,b])
        if len(combined)>72 or combined in (tuple(base),singles[a],singles[b]):continue
        result=execute(grid,combined)
        if result.error_reason==ErrorReason(reason) and goal_endpoint(grid,combined)==hard:return (a,b)
    return None
