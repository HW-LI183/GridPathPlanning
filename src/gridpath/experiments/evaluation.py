"""共同评价/CPU重放；无性能门槛，开发与两种各200实例测试使用同一协议。"""
from dataclasses import asdict
from collections import Counter
import csv,io,json,re,time
from ..training_config import read_json,write_json,sha256,digest,model_keys
from ..data_schema import grid_from_record
from ..executor import execute
from ..types import Grid
from ..bfs import shortest_path
from ..prompts import render_generation
from ..inference import Prediction,score_path
from ..evaluation import repair_once,Predictor
from ..baselines.feedback import repair_from_self,verification_report,summarize
from ..metrics import parse_verification_label,aggregate_verification_metrics
from .data import ROOT,MANIFEST,SPLITS,BRANCHES,load
from .config import completed_model,config_for,budget
from .ledger import RecordLedger
from .verifiers import prompt,validate


def directory_for(method,model,split):return ROOT/f'results/{method}/{model}/{split}'


def prompt_target(grid,text):
    match=re.search(r'Start \( (\d+) , (\d+) \)\. Goal \( (\d+) , (\d+) \)',text)
    if not match:raise ValueError('无法审计子问题输入')
    sr,sc,gr,gc=map(int,match.groups());ref=shortest_path(Grid(grid.rows,grid.cols,grid.obstacles,(sr,sc),(gr,gc)))
    return ' '.join(ref.actions) if ref.reachable else ''


def key_for(ledger,task,sid,branch=None):
    found=[k for k,r in ledger.slots.items() if r['task']==task and r['source_sample_id']==sid and (branch is None or r.get('branch')==branch)]
    if len(found)!=1:raise ValueError('任务ID不唯一')
    return found[0]


def joint_case(row,g,v,cfg,ledger):
    grid=grid_from_record(row);sid=row['sample_id'];trace=[];method=cfg['method']
    if row['task']=='GENERATE':
        text=render_generation(grid);key=key_for(ledger,'GENERATE',sid)
        initial=ledger.generation(key,text,' '.join(row['reference_plan']),g)
        trace.append({'prompt':text,'prediction':asdict(initial)})
    else:initial=Prediction(' '.join(row['candidate_plan']),0)
    text=prompt(method,grid,initial.text)
    pred,detail=ledger.verification(key_for(ledger,'VERIFY',sid),text,
        lambda:execute(grid,initial.text).verification_label,lambda:v(grid,initial.text))
    validate(method,initial.text,pred,detail);label=pred.text if pred.failure is None else ''
    outputs={}
    for branch in BRANCHES:
        repair_method=branch.split('_',1)[1];key=key_for(ledger,'REPAIR',sid,branch);calls=0
        def predictor(text):
            nonlocal calls
            calls+=1
            if calls>1:raise ValueError('单次修复最多一个模型输入，禁止隐式添加候选')
            p=ledger.generation(key,text,lambda:prompt_target(grid,text),g)
            trace.append({'prompt':text,'prediction':asdict(p)});return p
        final,info=(repair_from_self(grid,initial,label,repair_method,predictor) if branch.startswith('self_')
                    else repair_once(grid,initial,repair_method,predictor))
        if not calls:ledger.skip(key,info['status'])
        result,scoring=score_path(row,final,cfg['max_plan_actions'])
        outputs[branch]={'prediction':asdict(final),'detail':info,'scoring':scoring,'success':result.success,'feasible':result.feasible}
    original,scoring=score_path(row,initial,cfg['max_plan_actions'])
    result={'sample_id':sid,'row_digest':digest(row),'initial':asdict(initial),'initial_scoring':scoring,
       'initial_success':original.success,'executor_feedback':scoring['execution']['verification_label'],
       'self_verification':{'prompt':text,'prediction':asdict(pred),'effective_label':label,'detail':detail},
       'outcomes':outputs,'prediction_trace':trace,'feedback_model':method}
    return json.loads(json.dumps(result))


def fixed_case(row,v,cfg,ledger):
    grid=grid_from_record(row);text=prompt(cfg['method'],grid,row['candidate_plan'])
    pred,detail=ledger.verification(key_for(ledger,'VERIFY',row['sample_id']),text,row['target_text'],lambda:v(grid,row['candidate_plan']))
    validate(cfg['method'],row['candidate_plan'],pred,detail)
    return {'sample_id':row['sample_id'],'row_digest':digest(row),'executor_feedback':row['target_text'],
            'self_verification':{'prompt':text,'prediction':asdict(pred),'effective_label':pred.text if pred.failure is None else '', 'detail':detail}}


def subset_verification(rows,cases):
    selected=[c for r,c in zip(rows,cases,strict=True) if not r.get('is_valid',False) and r.get('unchecked_endpoint_is_goal',False)]
    detected=sum((p:=parse_verification_label(c['self_verification']['effective_label'])) is not None and not p.valid for c in selected)
    collisions=[c for c in selected if 'reason=obstacle_collision' in c['executor_feedback']]
    cd=sum((p:=parse_verification_label(c['self_verification']['effective_label'])) is not None and not p.valid for c in collisions)
    return {'count':len(selected),'detected':detected,'invalid_recall':detected/len(selected) if selected else None,
            'collision_count':len(collisions),'collision_detected':cd,'collision_recall':cd/len(collisions) if collisions else None,
            'metrics':verification_report(selected) if selected else None}


def collision_metrics(rows,cases):
    pairs=[(r,c) for r,c in zip(rows,cases,strict=True) if r['task']=='GENERATE' and
           c['initial_scoring']['execution']['error_reason']=='obstacle_collision']
    out={}
    for panel,subset in [('all_collisions',pairs),('endpoint_at_goal_collisions',[(r,c) for r,c in pairs if _endpoint_goal(r,c)])]:
        correct=detected=localized=0;missed=[]
        for _,c in subset:
            truth=parse_verification_label(c['executor_feedback']);p=parse_verification_label(c['self_verification']['effective_label'])
            detected+=p is not None and not p.valid
            localized+=p is not None and not p.valid and p.step==truth.step
            correct+=p is not None and not p.valid and p.step==truth.step and p.reason==truth.reason
            if p is None or p.valid:missed.append(c['sample_id'])
        out[panel]={'count':len(subset),'detected':detected,'joint_correct':correct,
           'invalid_recall':detected/len(subset) if subset else None,'localized':localized,
           'localization_accuracy':localized/len(subset) if subset else None,
           'joint_accuracy':correct/len(subset) if subset else None,'missed_ids':missed,
           'repairs':{b:sum(c['outcomes'][b]['success'] for _,c in subset) for b in BRANCHES}}
    return out


def _endpoint_goal(row,case):
    from ..planning.protocol import virtual_states
    return virtual_states(tuple(row['start']),case['initial']['text'])[-1]==tuple(row['goal'])


def report_for(model,cfg,data,split,cases,fixed,binding,ledger):
    G,V,R=(data[split+'_'+k] for k in ('generate','verify','repair_seeds'));rows=G+R
    report={'version':'gridpath.full_evaluation.1','method':cfg['method'],'model':model,'split':split,
      'fingerprint':binding['fingerprint'],'panels':{'generated':summarize(G,cases,cfg),'controlled_repair':summarize(R,cases,cfg)},
      'fixed_verification':verification_report(fixed),'verification_all_candidates':verification_report(cases),
      'fixed_hard_verification':subset_verification(V,fixed),'real_collision_verification':collision_metrics(rows,cases),
      'record_budget':ledger.audit(),'panel_sizes':{'generation':len(G),'fixed_verification':len(V),'controlled_repair_candidates':len(R)},
      'checkpoint_policy':'fixed_final_epoch_latest','test_used_for_selection':False}
    report['fixed_verifier_diagnostics']=report['fixed_verification']['self']
    if cfg['method']!='v2':
        from ..planning.verification import metrics
        report['structured_verifier_diagnostics']=metrics(V,[{'prediction':c['self_verification']['prediction'],
          'detail':c['self_verification']['detail']} for c in fixed])
    report['residual_error_transitions']={panel:{branch:dict(Counter(c['executor_feedback'].split('reason=')[-1]+'->'+
         ('valid' if c['outcomes'][branch]['success'] else c['outcomes'][branch]['scoring']['execution']['verification_label'])
         for c in cases if c['sample_id'] in {r['sample_id'] for r in group})) for branch in BRANCHES}
         for panel,group in [('generated',G),('controlled_repair',R)]}
    return report


def evaluate_with_predictors(model,cfg,data,split,g,v,directory,binding,save=True):
    ledger=RecordLedger(data[split+'_records'],directory/'task_records',binding['fingerprint'],data[split+'_samples'])
    fixed=[fixed_case(r,v,cfg,ledger) for r in data[split+'_verify']]
    cases=[joint_case(r,g,v,cfg,ledger) for r in data[split+'_generate']+data[split+'_repair_seeds']]
    report=report_for(model,cfg,data,split,cases,fixed,binding,ledger)
    if save:
        for c in fixed:write_json(directory/'verification_cases'/(c['sample_id']+'.json'),c)
        for c in cases:write_json(directory/'cases'/(c['sample_id']+'.json'),c)
        write_json(directory/'task_record_audit.json',report['record_budget']);write_json(directory/'summary.json',report)
    return report


def evaluation_binding(method,model,split):
    gcfg,gpath,gb=completed_model('v2',model,'generate')
    vcfg,vpath,vb=completed_model(method,model,'verify')
    cfg={**gcfg,'method':method}
    def files(path):return {k:v for k,v in read_json(path/'checkpoint.json')['files'].items() if k!='training_state.pt'}
    payload={'method':method,'model':model,'split':split,'G':gb['fingerprint'],'V':vb['fingerprint'],
             'G_files':files(gpath),'V_files':files(vpath),'dataset':sha256(MANIFEST),'protocol':'G60_V80_R60_four_routes_one_attempt'}
    return cfg,vcfg,{'fingerprint':digest(payload),**payload},gpath,vpath


def run(method,model,split='dev'):
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    from ..training_helpers import require_gpu,seed_everything
    if split not in SPLITS:raise ValueError('未知评价分区')
    if split!='dev':
        lock=ROOT/f'configs/{method}/primary_model.json'
        if not lock.exists() or read_json(lock)!=select_primary(method,False):raise ValueError('先完成所选模型开发评价并冻结选模与协议，测试不选模')
        if model not in [c['model'] for c in read_json(lock)['candidates']]:raise ValueError('模型不在本轮冻结的开发比较集合中')
    cfg,vcfg,binding,gpath,vpath=evaluation_binding(method,model,split)
    directory=directory_for(method,model,split)
    if (directory/'binding.json').exists() and read_json(directory/'binding.json')!=binding:raise ValueError('不能复用其他来源的推理记录')
    write_json(directory/'binding.json',binding)
    if (directory/'summary.json').exists():return read_json(directory/'summary.json')
    if split!='dev':
        marker=ROOT/f'manifests/test_usage_{method}.json'
        value={'method':method,'dataset':sha256(MANIFEST),'dev_selection':sha256(ROOT/f'configs/{method}/primary_model.json')}
        if marker.exists() and read_json(marker)!=value:raise ValueError('测试开始后不可改变开发选择或数据')
        write_json(marker,value)
    require_gpu(cfg)
    with budget(method,model,split) as resource:
        seed_everything(42);began=time.perf_counter()
        gt=AutoTokenizer.from_pretrained(gpath,local_files_only=True);vt=AutoTokenizer.from_pretrained(vpath,local_files_only=True)
        gn=AutoModelForSeq2SeqLM.from_pretrained(gpath,local_files_only=True).to('cuda').eval()
        if method=='v2':
            from .verifiers import PlainVerifier
            vn=AutoModelForSeq2SeqLM.from_pretrained(vpath,local_files_only=True).to('cuda').eval()
            v=PlainVerifier(vn,vt,vcfg,resource,directory/'verify_cache')
        else:
            from ..planning.model import StepVerifier
            from ..planning.verification import Verifier
            vn=StepVerifier.from_pretrained(vpath,local_files_only=True).to('cuda').eval()
            v=Verifier(vn,vt,vcfg,resource,directory/'verify_cache')
        g=Predictor(gn,gt,cfg,resource,directory/'generate_cache')
        data=load([split+'_'+k for k in ('generate','verify','repair_seeds','records','samples')])
        report=evaluate_with_predictors(model,cfg,data,split,g,v,directory,binding)
        report['performance']={'seconds':time.perf_counter()-began,'actual_generate_calls':g.calls,'actual_verify_calls':v.calls,
            'cache_hits':g.hits+v.hits,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        write_json(directory/'summary.json',report)
    return report


def summarize_files(method,split='dev',selected=None):
    selected=tuple(m for m in model_keys() if m in (selected or model_keys()))
    root=ROOT/f'results/{method}';models={m:read_json(directory_for(method,m,split)/'summary.json') for m in selected if (directory_for(method,m,split)/'summary.json').exists()}
    result={'method':method,'split':split,'complete':len(models)==len(selected),'requested_models':list(selected),'models':models}
    write_json(root/f'summary_{split}.json',result);paths=[];verification=[]
    for model,report in models.items():
        for panel,body in report['panels'].items():
            paths.append({'model':model,'panel':panel,'verifier':'offline_scoring','method':'direct',**body['direct']})
            for verifier,route in body['feedback_routes'].items():
                for repair,m in route['repairs'].items():paths.append({'model':model,'panel':panel,'verifier':verifier,'method':repair,**m})
            for verifier,m in body['verification'].items():verification.append({'model':model,'panel':panel,'verifier':verifier,**m})
        for panel,key in [('fixed_V80','fixed_verification'),('all_candidates','verification_all_candidates')]:
            for verifier,m in report[key].items():verification.append({'model':model,'panel':panel,'verifier':verifier,**m})
    for name,rows in [('metrics',paths),('verification',verification)]:
        columns=['model','panel','verifier']+(['method'] if name=='metrics' else [])
        columns+=sorted({k for r in rows for k,v in r.items() if k not in columns and not isinstance(v,(list,dict))})
        stream=io.StringIO();writer=csv.DictWriter(stream,columns,extrasaction='ignore');writer.writeheader()
        for r in rows:writer.writerow({k:v for k,v in r.items() if k in columns})
        (root/f'{name}_{split}.csv').write_text(stream.getvalue())
    return result


def select_primary(method,write=True,models=None):
    total=read_json(ROOT/f'results/{method}/summary_dev.json')
    if not total['complete']:raise ValueError('所选模型开发评价尚未完成')
    if models is not None and set(models)!=set(total['models']):raise ValueError('先按同一模型集合汇总开发结果')
    candidates=[]
    for m,r in total['models'].items():
        g=r['panels']['generated'];v=r['fixed_verification']['self']
        candidates.append({'model':m,'score':[g['direct']['success_rate'],v['macro_f1'],v['first_error_localization_accuracy'],
            max(g['feedback_routes']['self']['repairs'][k]['success_rate'] for k in ('prompt','prefix')),-model_keys().index(m)]})
    result={'method':method,'selected':max(candidates,key=lambda c:c['score']),'candidates':candidates,
        'selection':'仅dev连续指标排序，无性能门槛；G成功、V Macro-F1、定位、最佳self修复、固定模型顺序'}
    path=ROOT/f'configs/{method}/primary_model.json'
    if write:
        if path.exists() and read_json(path)!=result and (ROOT/f'manifests/test_usage_{method}.json').exists():
            raise ValueError('测试已经开始，不能改变开发选模或模型集合')
        write_json(path,result)
    return result


def review(method,split='dev',selected=None):
    total=read_json(ROOT/f'results/{method}/summary_{split}.json')
    if not total['complete']:raise ValueError('所选模型评价未完成')
    selected=tuple(m for m in model_keys() if m in (selected or model_keys()))
    if set(selected)!=set(total['models']):raise ValueError('复核模型集合与汇总不符')
    data=load([split+'_'+k for k in ('generate','verify','repair_seeds','records','samples')])
    def no_model(*args):raise AssertionError('CPU重放不得运行模型')
    for model in selected:
        cfg,_,binding,_,_=evaluation_binding(method,model,split);directory=directory_for(method,model,split)
        if read_json(directory/'binding.json')!=binding:raise ValueError('评价来源不符')
        ledger=RecordLedger(data[split+'_records'],directory/'task_records',binding['fingerprint'],data[split+'_samples'])
        for row in data[split+'_generate']+data[split+'_repair_seeds']:
            if joint_case(row,no_model,no_model,cfg,ledger)!=read_json(directory/'cases'/(row['sample_id']+'.json')):raise ValueError('修复案例重放不符')
        for row in data[split+'_verify']:
            if fixed_case(row,no_model,cfg,ledger)!=read_json(directory/'verification_cases'/(row['sample_id']+'.json')):raise ValueError('验证案例重放不符')
        expected=evaluate_with_predictors(model,cfg,data,split,no_model,no_model,directory,binding,False)
        if expected!={k:v for k,v in total['models'][model].items() if k!='performance'}:raise ValueError('汇总重算不符')
    result={'status':'passed','method':method,'split':split,'gpu_calls':0,'independent_samples_per_model':200}
    write_json(ROOT/f'results/{method}/review_{split}.json',result);return result


def compare(split='dev',selected=None):
    selected=tuple(m for m in model_keys() if m in (selected or model_keys()))
    a=read_json(ROOT/f'results/v2/summary_{split}.json');b=read_json(ROOT/f'results/improvement_verify_v1/summary_{split}.json')
    if not a['complete'] or not b['complete']:raise ValueError('需要完整基线和改进结果')
    result={'baseline':'v2','improvement':'improvement_verify_v1','split':split,'same_G_and_executor_outputs':True,'models':{}}
    for model in selected:
        if model not in a['models'] or model not in b['models']:raise ValueError('比较模型缺少完整结果：'+model)
        for path in (directory_for('v2',model,split)/'cases').glob('*.json'):
            x=read_json(path);y=read_json(directory_for('improvement_verify_v1',model,split)/'cases'/path.name)
            if x['initial']!=y['initial'] or any(x['outcomes'][k]!=y['outcomes'][k] for k in ('executor_prompt','executor_prefix')):raise ValueError('共用G或执行器对照发生变化')
        p,q=a['models'][model],b['models'][model]
        result['models'][model]={'fixed_verification':{'baseline':p['fixed_verification'],'improvement':q['fixed_verification']},
            'panels':{panel:{'baseline':p['panels'][panel],'improvement':q['panels'][panel]} for panel in ('generated','controlled_repair')}}
    write_json(ROOT/f'results/comparisons/verify_v1_vs_baseline_{split}.json',result);return result
