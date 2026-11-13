"""生成—验证—修复共有流程；朴素与结构化验证器共用相同修复及汇总逻辑。"""
from dataclasses import asdict
from collections import Counter
from ..data_schema import grid_from_record,grid_to_dict
from ..types import Grid,ACTION_DELTAS
from ..executor import execute,normalize_actions
from ..inference import Prediction,score_path
from ..prompts import render_generation,render_repair
from ..metrics import parse_verification_label,aggregate_verification_metrics
from ..evaluation import summarize_panel

def repair_from_self(grid,original,label,method,predict):
    """只用模型诊断。机械推算所声称前缀的终点，不执行/截断到真首错。"""
    parsed=parse_verification_label(label)
    detail={'feedback':label,'model_calls':0,'feedback_source':'self'}
    if parsed is None:return original,{**detail,'status':'verification_format_failure'}
    if parsed.valid:return original,{**detail,'status':'self_valid_preserved'}
    if method=='prompt':
        prompt=render_repair(grid,original.text,label);output=predict(prompt)
        return output,{**detail,'status':'prompt_repair','model_calls':1,'prompt':prompt,'output':asdict(output)}
    if method!='prefix':raise ValueError('未知修复方法')
    actions=normalize_actions(original.text)
    if not 1<=parsed.step<=len(actions)+1:return original,{**detail,'status':'self_prefix_step_out_of_range'}
    prefix=actions[:parsed.step-1];position=grid.start
    for action in prefix:
        if action not in ACTION_DELTAS:return original,{**detail,'status':'self_prefix_unknown_action'}
        dr,dc=ACTION_DELTAS[action];position=(position[0]+dr,position[1]+dc)
    # 不检测沿途碰撞，不用executor纠正模型首错。仅保证子问题端点可被序列化。
    detail.update(prefix=list(prefix),claimed_step=parsed.step)
    try:subgrid=Grid(grid.rows,grid.cols,grid.obstacles,position,grid.goal)
    except ValueError:return original,{**detail,'status':'self_prefix_endpoint_unusable'}
    detail['subproblem']=grid_to_dict(subgrid)
    if position==grid.goal:return Prediction(' '.join(prefix),0),{**detail,'status':'empty_suffix_at_goal'}
    prompt=render_generation(subgrid);suffix=predict(prompt)
    output=Prediction(' '.join((*prefix,*normalize_actions(suffix.text))),suffix.generated_tokens,suffix.failure)
    return output,{**detail,'status':'prefix_regeneration','model_calls':1,'prompt':prompt,'suffix':asdict(suffix)}

def verification_report(cases):
    truth=[c['executor_feedback'] for c in cases]
    predicted=[c['self_verification']['effective_label'] for c in cases]
    model=aggregate_verification_metrics(truth,predicted)
    model['logical_verification_model_calls']=len(cases)
    model['interface_failure_count']=sum(c['self_verification']['prediction']['failure'] is not None for c in cases)
    confusion=Counter()
    for t,p in zip(truth,predicted):
        tp=parse_verification_label(t);pp=parse_verification_label(p)
        confusion[('valid' if tp.valid else 'invalid')+'->'+('format_error' if pp is None else 'valid' if pp.valid else 'invalid')]+=1
    model['confusion']=dict(confusion)
    return {'self':model,'executor':aggregate_verification_metrics(truth,truth)}

def summarize(rows,cases,config):
    cases_by_id={c['sample_id']:c for c in cases};ordered=[cases_by_id[r['sample_id']] for r in rows]
    result={'count':len(rows),'verification':verification_report(ordered),'feedback_routes':{}}
    for verifier in ('executor','self'):
        translated=[{**c,'outcomes':{m:c['outcomes'][f'{verifier}_{m}'] for m in ('prompt','prefix')}} for c in ordered]
        body=summarize_panel(rows,translated,config)
        result['direct']=body['direct']
        result['feedback_routes'][verifier]=body
    for route in result['feedback_routes'].values():
        for metrics in route['repairs'].values():
            metrics['logical_verification_model_calls']=len(rows) if route is result['feedback_routes']['self'] else 0
    return result
