"""语言模型验证推理及离线评分；predict不调用执行器或真值。"""
from collections import Counter
import time
import torch
from ..training_config import digest,read_json,write_json
from ..inference import Prediction,autocast_context
from ..data_schema import grid_from_record
from ..metrics import aggregate_verification_metrics,parse_verification_label
from .protocol import feature,aggregate,TAGS,actions_of
from .model import collate,path_loss


class Verifier:
    def __init__(self,model,tokenizer,cfg,budget=None,cache=None):
        self.model,self.tokenizer,self.cfg,self.budget,self.cache=model,tokenizer,cfg,budget,cache
        self.device=next(model.parameters()).device
        self.calls=self.hits=0;self.seconds=0.

    def __call__(self,grid,candidate):
        actions=actions_of(candidate)
        key=digest({'grid':{'size':[grid.rows,grid.cols],'start':grid.start,'goal':grid.goal,
                           'obstacles':sorted(grid.obstacles)},'candidate':actions})
        path=self.cache/(key+'.json') if self.cache else None
        if path and path.exists():
            saved=read_json(path)
            if saved['key']!=key:raise ValueError('验证缓存键不符')
            self.hits+=1
            return Prediction(**saved['prediction']),saved['detail']
        if self.budget:self.budget.check(reserve_seconds=30)
        try:
            if len(actions)>self.cfg['max_plan_actions']:
                raise ValueError('候选超过动作上限')
            f=feature(self.tokenizer,grid,actions,self.cfg['verify_source_tokens'])
        except ValueError as exc:
            return Prediction('',0,'verifier_source_limit_or_marker'),{'failure':str(exc),'tags':[]}
        inputs,_=collate([f],self.tokenizer.pad_token_id,self.device)
        self.model.eval()
        if self.device.type=='cuda':torch.cuda.synchronize()
        began=time.perf_counter()
        with torch.inference_mode(),autocast_context(self.device,self.cfg['precision']):
            logits=self.model(**inputs).float()[0]
            if not torch.isfinite(logits).all():raise FloatingPointError('验证logits非有限')
            tags=logits.argmax(-1).tolist();probabilities=logits.softmax(-1).cpu().tolist()
        if self.device.type=='cuda':torch.cuda.synchronize()
        self.seconds+=time.perf_counter()-began;self.calls+=1
        output=Prediction(aggregate(tags,len(actions)),0)
        detail={'tags':tags,'tag_names':[TAGS[t] for t in tags],
                'probabilities':probabilities,'input_tokens':len(f['input_ids']),
                'feedback_source':'language_model_step_tagging','executor_used':False}
        if path:write_json(path,{'key':key,'prediction':output.__dict__,'detail':detail})
        return output,detail


def metrics(rows,predictions):
    labels=[r['target_text'] for r in rows]
    texts=[p['prediction']['text'] if p['prediction']['failure'] is None else '' for p in predictions]
    m=aggregate_verification_metrics(labels,texts)
    modes=Counter()
    for p in predictions:
        parsed=parse_verification_label(p['prediction']['text'])
        if parsed and not parsed.valid:modes[parsed.step]+=1
    m['predicted_error_steps']={str(k):v for k,v in sorted(modes.items())}
    m['predicted_error_types']=dict(Counter(
        parsed.reason.value for text in texts
        if (parsed:=parse_verification_label(text)) is not None and not parsed.valid))
    m['tag_confusion']=dict(Counter(
        TAGS[truth]+'->'+TAGS[pred]
        for row,p in zip(rows,predictions,strict=True)
        for truth,pred in zip(row['trace_labels'],p['detail']['tags']) if truth!=-100))
    true_invalid=[(parse_verification_label(t),parse_verification_label(p)) for t,p in zip(labels,texts)
                  if not parse_verification_label(t).valid]
    later=[(t,p) for t,p in true_invalid if t.step>1]
    # step=1是已有训练集位置众数；同时比较同一模型分类门控下的常量位置。
    n=len(true_invalid)
    m['constant_step1_reference']={
        'invalid_count':n,
        'unconditional_accuracy':sum(t.step==1 for t,_ in true_invalid)/n if n else None,
        'classification_conditioned_accuracy':sum(t.step==1 and p is not None and not p.valid for t,p in true_invalid)/n if n else None}
    m['later_error_localization']={'count':len(later),'correct_count':sum(p is not None and not p.valid and p.step==t.step for t,p in later)}
    m['interface_failure_count']=sum(p['prediction']['failure'] is not None for p in predictions)
    m['by_candidate_length']={}
    for group,indices in {'0-4':[i for i,r in enumerate(rows) if len(r['candidate_plan'])<=4],
                          '5-7':[i for i,r in enumerate(rows) if 5<=len(r['candidate_plan'])<=7],
                          '8+':[i for i,r in enumerate(rows) if len(r['candidate_plan'])>=8]}.items():
        if indices:m['by_candidate_length'][group]=aggregate_verification_metrics([labels[i] for i in indices],[texts[i] for i in indices])
    return m


def report(model,tokenizer,rows,cfg,budget=None):
    verifier=Verifier(model,tokenizer,cfg,budget)
    predictions=[];loss=0.
    for row in rows:
        grid=grid_from_record(row);pred,detail=verifier(grid,row['candidate_plan'])
        predictions.append({'sample_id':row['sample_id'],'prediction':pred.__dict__,'detail':detail})
        f=feature(tokenizer,grid,row['candidate_plan'],cfg['verify_source_tokens'],row['trace_labels'])
        inputs,labels=collate([f],tokenizer.pad_token_id,verifier.device)
        with torch.inference_mode(),autocast_context(verifier.device,cfg['precision']):
            value=path_loss(model(**inputs),labels)
        if not torch.isfinite(value):raise FloatingPointError('开发损失非有限')
        loss+=float(value)
    return {'count':len(rows),'metrics':metrics(rows,predictions),'nll':loss/len(rows),
            'predictions':predictions,'performance':{'encoder_calls':verifier.calls,
                'seconds':verifier.seconds,'generated_tokens':0}}

