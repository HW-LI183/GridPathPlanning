"""中立数据协议：G3000/V1000；开发、单错、双错各G60/V80/R60。"""
from pathlib import Path
from collections import Counter
import json,random,time
from ..training_config import read_json,read_rows,write_json,sha256,digest
from ..data_schema import grid_from_record,grid_to_dict,layout_id,layout_split,environment_id
from ..types import Grid,ErrorReason
from ..bfs import shortest_path
from ..executor import execute
from ..prompts import render_generation
from ..v2.prompts import render_verification
from ..planning.events import choices,apply_events,safe_loop,double_events,event_from_dict,goal_endpoint
from ..planning.protocol import TAGS

ROOT=Path(__file__).resolve().parents[3]
DIRECTORY=ROOT/'data/v2/shared'
MANIFEST=ROOT/'manifests/dataset_v2.json'
SPLITS=('dev','test_id','test_ood')
BRANCHES=('self_prompt','self_prefix','executor_prompt','executor_prefix')
SPEC_NAMES=('train_generate','train_verify','dev_generate','dev_verify','dev_repair_seeds')
# 四种无效类别等量；未到目标没有硬负例，其余三类各取2/3，
# 因此硬负例占全部无效候选最多1/2，整数配额向下取整。
HARD_FRACTION=(2,3)


def hard_quota(count):
    return count*HARD_FRACTION[0]//HARD_FRACTION[1]


def rebalance_negatives(rows):
    """保留冻结环境、参考路径、类别和ID，将部分硬负例改为普通单事件负例。"""
    rows=[dict(r) for r in rows]
    for reason in ErrorReason:
        if reason==ErrorReason.GOAL_NOT_REACHED:continue
        group=[r for r in rows if r.get('error_reason')==reason.value]
        hard=sorted([r for r in group if r['unchecked_endpoint_is_goal']],key=lambda r:digest([42,r['sample_id']]))
        if len(hard)<hard_quota(len(group)):raise ValueError('蓝图硬负例不足，不能隐式扩大配额')
        for row in hard[hard_quota(len(group)):]:
            grid=grid_from_record(row);base=row['base_plan']
            pool=[e for e in choices(grid,base,reason) if not goal_endpoint(grid,apply_events(base,[e]))]
            if not pool:raise ValueError('同环境无对应类别普通负例：'+row['sample_id'])
            e=min(pool,key=lambda e:digest([42,'ordinary',row['sample_id'],e.to_dict()]))
            candidate=list(apply_events(base,[e]))
            row.update(candidate_plan=candidate,error_events=[e.to_dict()],atomic_error_count=1,
                       primitive_token_edits=e.primitive_token_edits,single_counterfactual_candidates=[candidate],
                       unchecked_endpoint_is_goal=False)
    return [canonical_row(r) for r in rows]


def trace_labels(grid,candidate):
    result=execute(grid,candidate);labels=[-100]*(len(result.actions)+1)
    for step in result.full_trace:
        labels[step.step-1]=TAGS.index(step.error_reason.value) if step.error_reason else 0
    if result.success:labels[-1]=TAGS.index('valid_end')
    return labels


def canonical_row(row,split=None):
    row={**row,'version':'gridpath.task.v2','split':split or row['split']}
    grid=grid_from_record(row)
    if row['task']=='GENERATE':
        row.update(input_text=render_generation(grid),target_text=' '.join(row['reference_plan']))
    else:
        truth=execute(grid,row['candidate_plan'])
        row.update(input_text=render_verification(grid,row['candidate_plan']),target_text=truth.verification_label,
            verification_label=truth.verification_label,is_valid=truth.success,
            error_reason=truth.error_reason.value if truth.error_reason else None,
            trace_labels=trace_labels(grid,row['candidate_plan']))
    return row


def new_row(grid,split,task,candidate=None,events=(),positive_event=None):
    ref=shortest_path(grid)
    row={'split':split,'task':task,**grid_to_dict(grid),'layout_id':layout_id(grid),
         'environment_id':environment_id(grid),'reference_plan':list(ref.actions),'shortest_distance':ref.distance}
    row['paired_id']='pair_'+digest([task,row['environment_id']])[:24]
    if candidate is not None:
        row.update(base_plan=list(ref.actions),candidate_plan=list(candidate),error_events=[e.to_dict() for e in events],
            atomic_error_count=len(events),primitive_token_edits=sum(e.primitive_token_edits for e in events),
            single_counterfactual_candidates=[list(apply_events(ref.actions,[e])) for e in events],
            positive_construction=positive_event.to_dict() if positive_event else None,
            unchecked_endpoint_is_goal=goal_endpoint(grid,candidate))
    row['sample_id']='sample_'+digest([split,task,row['environment_id'],candidate])[:24]
    return canonical_row(row)


def make_tests(train,dev):
    rng=random.Random(4202026);used={r['environment_id'] for r in train+dev}
    forbidden={r['layout_id'] for r in train+dev};cells=[(r,c) for r in range(6) for c in range(6)]
    panels={s:{'generate':[],'verify':[],'repair_seeds':[]} for s in ('test_id','test_ood')}
    def sample(i):
        for _ in range(20000):
            blocked=frozenset(rng.sample(cells,i%5+1));free=[p for p in cells if p not in blocked]
            start,goal=rng.sample(free,2);g=Grid(6,6,blocked,start,goal)
            if layout_id(g) in forbidden or layout_split(layout_id(g),42)!='test' or environment_id(g) in used:continue
            ref=shortest_path(g)
            if ref.reachable and 2<=ref.distance<=72:return g,ref
        raise RuntimeError('测试环境有界采样超限')
    def reserve(g):used.add(environment_id(g))
    for i in range(60):
        g,_=sample(i);reserve(g)
        for s in panels:panels[s]['generate'].append(new_row(g,s,'GENERATE'))
    for i in range(40):
        g,ref=sample(i);reserve(g);candidate=ref.actions;event=None
        if i<16:candidate,event=safe_loop(g,ref.actions,environment_id(g))
        for s in panels:panels[s]['verify'].append(new_row(g,s,'VERIFY',candidate,positive_event=event))
    for task,key,n in [('VERIFY','verify',10),('REPAIR','repair_seeds',15)]:
        for reason in ErrorReason:
            for j in range(n):
                hard=reason!=ErrorReason.GOAL_NOT_REACHED and j<hard_quota(n)
                for attempt in range(400):
                    g,ref=sample(j+list(ErrorReason).index(reason));identity=[task,reason.value,j,environment_id(g)]
                    kind=('insert_invalid_token' if reason==ErrorReason.INVALID_ACTION_TOKEN else 'return_block') if hard else None
                    pool=[e for e in choices(g,ref.actions,reason,kind) if goal_endpoint(g,apply_events(ref.actions,[e]))==hard]
                    if not pool:continue
                    single=min(pool,key=lambda e:digest([42,identity,e.to_dict()]))
                    double=double_events(g,ref.actions,reason,hard,identity)
                    if double is None:continue
                    reserve(g)
                    panels['test_id'][key].append(new_row(g,'test_id',task,apply_events(ref.actions,[single]),[single]))
                    panels['test_ood'][key].append(new_row(g,'test_ood',task,apply_events(ref.actions,double),double))
                    break
                else:raise RuntimeError('无法满足单错/双错配额，不降低要求：'+str(identity))
    for split in panels:
        for key in panels[split]:panels[split][key].sort(key=lambda r:digest([42,r['sample_id']]))
    return panels


def call_slots(split,g,v,r):
    slots=[]
    def add(task,key,row,**extra):slots.append({'record_id':f'{split}:{key}:{row["sample_id"]}',
        'task':task,'source_sample_id':row['sample_id'],'independent_sample_id':row['sample_id'],**extra})
    for row in g:add('GENERATE','G',row)
    for row in g+v+r:add('VERIFY','V',row)
    for row in g+r:
        for branch in BRANCHES:add('REPAIR',branch,row,branch=branch,derived=True)
    assert len(slots)==740
    return slots


def _write(name,rows):
    p=DIRECTORY/(name+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
    if p.exists() and p.read_text()!=text:raise ValueError('冻结数据不同，拒绝覆盖：'+name)
    p.write_text(text)
    return {'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'count':len(rows)}


def build():
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'configs/experiment.yaml']+
             [ROOT/f'data/specs/{n}.jsonl' for n in SPEC_NAMES]+
             [ROOT/f'src/gridpath/{n}.py' for n in ('executor','bfs','types','data_schema','corruptions','prompts')]+
             [ROOT/f'src/gridpath/{n}' for n in ('planning/events.py','planning/protocol.py','v2/prompts.py')]}
    if MANIFEST.exists():
        old=read_json(MANIFEST)
        if old['sources']!=sources:raise ValueError('数据协议变化，不能沿用已冻结数据')
        for f in old['files'].values():
            if sha256(ROOT/f['path'])!=f['sha256']:raise ValueError('冻结数据损坏')
        return old
    began=time.perf_counter();groups={n:[canonical_row(r) for r in read_rows(ROOT/f'data/specs/{n}.jsonl')] for n in SPEC_NAMES}
    for name in ('train_verify','dev_verify','dev_repair_seeds'):groups[name]=rebalance_negatives(groups[name])
    dev=sum((groups['dev_'+k] for k in ('generate','verify','repair_seeds')),[])
    tests=make_tests(groups['train_generate']+groups['train_verify'],dev)
    for s,panels in tests.items():
        for key,rows in panels.items():groups[s+'_'+key]=rows
    for s in SPLITS:
        g,v,r=(groups[s+'_'+k] for k in ('generate','verify','repair_seeds'))
        groups[s+'_samples']=g+v+r;groups[s+'_records']=call_slots(s,g,v,r)
    files={n:_write(n,rows) for n,rows in groups.items()}
    m={'version':'phase3.v2.full','sources':sources,'files':files,'train':{'GENERATE':3000,'VERIFY':1000,'REPAIR':0},
       'split_panels':{'GENERATE':60,'VERIFY':80,'REPAIR':60},'single_error_atomic_events':1,'multi_error_atomic_events':2,
       'test_pairing':'相同环境和参考路径；正例与G相同，负例分别1/2独立原子事件，不强制ID事件为OOD子集',
       'hard_fraction_within_eligible_error_classes':list(HARD_FRACTION),'gpu_calls':0}
    write_json(MANIFEST,m)
    write_json(ROOT/'manifests/data_construction_v2.json',{'seconds':time.perf_counter()-began,'gpu_calls':0})
    return m


def load(names=None):
    m=build()
    selected=m['files'] if names is None else {name:m['files'][name] for name in names}
    return {name:read_rows(ROOT/f['path']) for name,f in selected.items()}


def audit():
    d=load();g,v=d['train_generate'],d['train_verify'];assert len(g)==3000 and len(v)==1000
    assert Counter(r['is_valid'] for r in v)=={True:500,False:500}
    assert Counter(r['error_reason'] for r in v if not r['is_valid'])=={e.value:125 for e in ErrorReason}
    assert len({r['sample_id'] for r in g+v})==4000
    for reason in ErrorReason:
        if reason!=ErrorReason.GOAL_NOT_REACHED:
            assert sum(r.get('error_reason')==reason.value and r['unchecked_endpoint_is_goal'] for r in v)==hard_quota(125)
    assert all(r['environment_id'] in {x['environment_id'] for x in g} for r in v)
    for r in g:
        ref=shortest_path(grid_from_record(r))
        assert ref.reachable and ref.distance==r['shortest_distance']==len(r['reference_plan'])
        assert execute(grid_from_record(r),r['reference_plan']).success
    train_layout={r['layout_id'] for r in g+v};dev_layout={r['layout_id'] for r in d['dev_samples']}
    assert not train_layout&dev_layout
    for split in SPLITS:
        G,V,R=(d[split+'_'+k] for k in ('generate','verify','repair_seeds'))
        assert (len(G),len(V),len(R))==(60,80,60)
        assert len({r['sample_id'] for r in G+V+R})==len({r['environment_id'] for r in G+V+R})==200
        assert Counter(r['is_valid'] for r in V)=={True:40,False:40}
        assert Counter(r['error_reason'] for r in V if not r['is_valid'])=={e.value:10 for e in ErrorReason}
        assert Counter(r['error_reason'] for r in R)=={e.value:15 for e in ErrorReason}
        for rows,count in ((V,10),(R,15)):
            for reason in ErrorReason:
                if reason!=ErrorReason.GOAL_NOT_REACHED:
                    assert sum(r.get('error_reason')==reason.value and r['unchecked_endpoint_is_goal'] for r in rows)==hard_quota(count)
        if split!='dev':assert not (train_layout|dev_layout)&{r['layout_id'] for r in G+V+R}
        for row in V+R:
            grid=grid_from_record(row);truth=execute(grid,row['candidate_plan'])
            assert truth.verification_label==row['target_text'] and row['trace_labels']==trace_labels(grid,row['candidate_plan'])
            if truth.success:continue
            events=[event_from_dict(e) for e in row['error_events']]
            assert list(apply_events(row['base_plan'],events))==row['candidate_plan']
            expected=2 if split=='test_ood' else 1
            assert len(events)==row['atomic_error_count']==expected
            if expected==2:
                assert len({e.index for e in events})==2
                for e in events:assert not execute(grid,apply_events(row['base_plan'],[e])).success
                assert all(row['candidate_plan']!=list(apply_events(row['base_plan'],[e])) for e in events)
    for key in ('generate','verify','repair_seeds'):
        a={r['paired_id']:r for r in d['test_id_'+key]};b={r['paired_id']:r for r in d['test_ood_'+key]}
        assert a.keys()==b.keys()
        assert all(a[k]['environment_id']==b[k]['environment_id'] and a[k]['reference_plan']==b[k]['reference_plan'] for k in a)
    report={'status':'passed','train_records':4000,'dev_records':200,'test_id_records':200,'test_ood_records':200,
            'split_panels':{'G':60,'V':80,'R':60},'model_inputs_per_split':740,'gpu_calls':0,
            'single_atomic_events':1,'multi_atomic_events':2,'paired_test_environments':200}
    report['hard_negatives']={}
    for key in ('train_verify',)+tuple(s+'_'+k for s in SPLITS for k in ('verify','repair_seeds')):
        negatives=[r for r in d[key] if not r['is_valid']]
        count=sum(r['unchecked_endpoint_is_goal'] for r in negatives)
        assert count<=len(negatives)/2
        report['hard_negatives'][key]={'count':count,'invalid_count':len(negatives),'fraction':count/len(negatives)}
    write_json(ROOT/'manifests/data_audit_v2.json',report);return report
