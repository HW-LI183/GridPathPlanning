"""从原始检查点训练G/两种V，固定三遍末权重；没有性能门槛或best搜索。"""
import json,signal,time
from ..training_config import read_json,write_json
from ..training_helpers import (seed_everything,stream_indices,train_update,save_checkpoint,
    validate_checkpoint,load_training_state,require_gpu)
from ..inference import autocast_context
from ..data_schema import grid_from_record
from ..evaluation import Predictor,path_report
from ..metrics import aggregate_verification_metrics
from .data import ROOT
from .config import binding_for,budget


def state_check(state,progress,cfg,count):
    step,consumed=state['global_step'],state['consumed_examples']
    if consumed!=step*4 or step>cfg['max_updates'] or consumed>3*count or any(s not in cfg['evaluate_steps'] or s>step for s in state['evaluated_steps']):
        raise ValueError('训练暴露或更新超出三遍上限')
    if progress!={'reserved_examples':consumed,'completed_updates':step}:raise RuntimeError('存在未提交暴露，拒绝自动重放；请保留现场核查')


def structured_update(model,optimizer,scheduler,features,tokenizer,cfg):
    import torch
    from ..planning.model import collate,path_loss
    model.train();optimizer.zero_grad(set_to_none=True)
    inputs,labels=collate(features,tokenizer.pad_token_id,next(model.parameters()).device)
    with autocast_context(next(model.parameters()).device,cfg['precision']):loss=path_loss(model(**inputs),labels)
    if not torch.isfinite(loss):raise FloatingPointError('训练loss非有限')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return {'loss':float(loss.detach()),'grad_norm':float(norm),'learning_rate':scheduler.get_last_lr()[0]}


def evaluate_dev(network,tok,rows,cfg,resource):
    if cfg['role']=='generate':
        predictor=Predictor(network,tok,cfg,resource)
        predictions=[predictor(r['input_text']) for r in rows]
        metrics,_=path_report(rows,predictions,cfg)
        return {'count':len(rows),'metrics':metrics,'predictions':[p.__dict__ for p in predictions]}
    if cfg['structured']:
        from ..planning.verification import Verifier,metrics
        v=Verifier(network,tok,cfg,resource)
        predictions=[{'sample_id':r['sample_id'],'prediction':p.__dict__,'detail':detail}
                     for r in rows for p,detail in [v(grid_from_record(r),r['candidate_plan'])]]
        return {'count':len(rows),'metrics':metrics(rows,predictions),'predictions':predictions}
    predictor=Predictor(network,tok,cfg,resource);ps=[predictor(r['input_text']) for r in rows]
    return {'count':len(rows),'metrics':aggregate_verification_metrics([r['target_text'] for r in rows],
        [p.text if p.failure is None else '' for p in ps]),'predictions':[p.__dict__ for p in ps]}


def evaluate_pending(directory,network,tok,dev,cfg,resource,state,binding,checkpoint):
    step=state['global_step']
    if step not in cfg['evaluate_steps']:return
    path=directory/f'evaluations/step_{step:04d}.json'
    if path.exists():
        saved=read_json(path)
        if saved['fingerprint']!=binding['fingerprint'] or saved['step']!=step:raise ValueError('开发诊断来源或更新步不符')
        if step in state['evaluated_steps']:return
    else:
        report=evaluate_dev(network,tok,dev,cfg,resource)
        write_json(path,{'fingerprint':binding['fingerprint'],'step':step,**report})
    # 报告原子落盘后才标记完成；兼容有完成标记但报告写入失败的中断现场。
    if step not in state['evaluated_steps']:state['evaluated_steps'].append(step)
    checkpoint(['latest'])


def run(method,model,role='verify',pilot=False):
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    cfg,train,dev,binding=binding_for(method,model,role,verify_weights=True)
    directory=ROOT/cfg['run_directory'];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'run.json').exists() and read_json(directory/'run.json')!=binding:raise ValueError('已有不同训练身份，拒绝续用')
    write_json(directory/'run.json',binding)
    if (directory/'completed.json').exists():
        from .config import completed_model
        completed_model(method,model,role);return read_json(directory/'completed.json')
    if pilot and (directory/'pilot_report.json').exists():return read_json(directory/'pilot_report.json')
    require_gpu(cfg)
    with budget(method,model,role+(':pilot' if pilot else ':train')) as resource:
        seed_everything(42);began=time.perf_counter()
        tok=AutoTokenizer.from_pretrained(ROOT/cfg['model_directory'],local_files_only=True)
        if cfg['structured']:
            from ..planning.model import StepVerifier
            from ..planning.protocol import feature
            network=StepVerifier.from_backbone(ROOT/cfg['model_directory']).to('cuda')
            features=[feature(tok,grid_from_record(r),r['candidate_plan'],cfg['verify_source_tokens'],r['trace_labels']) for r in train]
        else:
            network=AutoModelForSeq2SeqLM.from_pretrained(ROOT/cfg['model_directory'],local_files_only=True).to('cuda')
            features=[{'input_ids':tok(r['input_text'],truncation=False)['input_ids'],
                       'labels':tok(r['target_text'],truncation=False)['input_ids']} for r in train]
            if any(len(f['input_ids'])>cfg['max_source_tokens'] or len(f['labels'])>cfg['max_target_tokens'] for f in features):raise ValueError('训练输入/目标超长，不截断')
        groups={}
        for name,p in network.named_parameters():
            lr=cfg['head_learning_rate'] if cfg['structured'] and name.startswith('classifier.') else cfg['learning_rate']
            decay=0. if name.endswith('bias') or 'layer_norm' in name else cfg['weight_decay']
            groups.setdefault((lr,decay),[]).append(p)
        opt=torch.optim.AdamW([{'params':ps,'lr':lr,'weight_decay':wd} for (lr,wd),ps in groups.items()])
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(step/cfg['warmup_updates'],1.))
        if (directory/'latest').exists():
            meta=validate_checkpoint(directory/'latest',binding['fingerprint']);state=meta['state']
            state_check(state,read_json(directory/'progress.json'),cfg,len(train))
            from safetensors.torch import load_model
            load_model(network,str(directory/'latest/model.safetensors'),strict=True,device='cuda')
            if load_training_state(directory/'latest',opt,scheduler,'cuda')!=state:raise ValueError('优化器/RNG状态不符')
        else:
            if (directory/'progress.json').exists():raise ValueError('已有暴露但没有快照，拒绝重放')
            state={'global_step':0,'consumed_examples':0,'evaluated_steps':[],'role':role}
            write_json(directory/'progress.json',{'reserved_examples':0,'completed_updates':0})
            save_checkpoint(directory,['latest'],network,tok,opt,scheduler,state,binding)
        if not pilot and (directory/'pilot').is_symlink():(directory/'pilot').unlink()
        checkpoint=lambda aliases:save_checkpoint(directory,aliases,network,tok,opt,scheduler,state,binding)
        def evaluate_current():
            evaluate_pending(directory,network,tok,dev,cfg,resource,state,binding,checkpoint)
        stopped={'value':False}
        def request_stop(*args):stopped['value']=True
        handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
        target=cfg['pilot_updates'] if pilot else cfg['max_updates']
        try:
            evaluate_current()
            while state['global_step']<target:
                if stopped['value'] or resource.remaining_seconds<90:
                    checkpoint(['latest']);return {'status':'stopped','state':state}
                indices=stream_indices(len(train),42,state['consumed_examples'],4)
                write_json(directory/'progress.json',{'reserved_examples':state['consumed_examples']+len(indices),'completed_updates':state['global_step']})
                update=(structured_update if cfg['structured'] else train_update)(network,opt,scheduler,[features[i] for i in indices],tok,cfg)
                state['global_step']+=1;state['consumed_examples']+=len(indices)
                write_json(directory/'progress.json',{'reserved_examples':state['consumed_examples'],'completed_updates':state['global_step']})
                with (directory/'training.jsonl').open('a') as f:f.write(json.dumps({'step':state['global_step'],'consumed':state['consumed_examples'],**update})+'\n')
                if state['global_step']%cfg['checkpoint_every_updates']==0:checkpoint(['latest'])
                evaluate_current()
                if state['global_step']%100==0:print(f'{method}/{model}/{role}: {state["global_step"]}/{cfg["max_updates"]}',flush=True)
            state_check(state,read_json(directory/'progress.json'),cfg,len(train))
            checkpoint(['latest','pilot'] if pilot else ['latest'])
            result={'status':'pilot_complete' if pilot else 'complete','fingerprint':binding['fingerprint'],
                    'state':state,'seconds':time.perf_counter()-began,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            write_json(directory/('pilot_report.json' if pilot else 'completed.json'),result);return result
        except BaseException:
            if read_json(directory/'progress.json')=={'reserved_examples':state['consumed_examples'],'completed_updates':state['global_step']}:checkpoint(['latest'])
            raise
        finally:
            for s,h in handlers.items():signal.signal(s,h)
