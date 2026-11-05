"""阶段四v1：执行器验证、单次提示修复与前缀续写，以及细粒度指标。"""
from collections import Counter
from dataclasses import asdict
import time

from .data_schema import grid_from_record,grid_to_dict
from .executor import execute,normalize_actions
from .inference import Prediction,score_path
from .metrics import aggregate_path_metrics,aggregate_repair_metrics,aggregate_verification_metrics
from .prompts import render_generation,render_repair
from .training_config import digest,read_json,write_json
from .types import Grid
from .types import ACTION_ORDER


class Predictor:
    def __init__(self,model,tokenizer,config,budget=None,cache=None):
        self.model,self.tokenizer,self.config,self.budget,self.cache=model,tokenizer,config,budget,cache
        self.calls=self.hits=self.tokens=0
        self.seconds=0.
        self.device=next(model.parameters()).device

    def __call__(self,text):
        import torch
        from transformers import StoppingCriteriaList
        from .inference import autocast_context,decode_prediction,BudgetStoppingCriteria
        path=self.cache/f"{digest(text)}.json" if self.cache else None
        if path and path.exists():
            cached=read_json(path)
            if cached["prompt"]!=text:raise ValueError("推理缓存不一致")
            self.hits+=1
            return Prediction(**cached["prediction"])
        if self.budget:self.budget.check(reserve_seconds=35)
        encoded=self.tokenizer(text,truncation=False,return_tensors="pt")
        if encoded.input_ids.shape[1]>self.config["max_source_tokens"]:
            result=Prediction("",0,"source_limit")
        else:
            encoded={k:v.to(self.device) for k,v in encoded.items()}
            self.model.eval()
            if self.device.type=="cuda":torch.cuda.synchronize()
            began=time.perf_counter()
            extra={"stopping_criteria":StoppingCriteriaList([BudgetStoppingCriteria(self.budget)])} if self.budget else {}
            with torch.inference_mode(),autocast_context(self.device,self.config["precision"]):
                output=self.model.generate(**encoded,max_new_tokens=self.config["generation_tokens"],
                    do_sample=False,num_beams=1,forced_eos_token_id=None,use_cache=True,
                    no_repeat_ngram_size=0,repetition_penalty=1.,**extra)
            if self.device.type=="cuda":torch.cuda.synchronize()
            self.seconds+=time.perf_counter()-began
            if self.budget:self.budget.check(reserve_seconds=30)
            result=decode_prediction(self.tokenizer,output[0].tolist(),self.config["generation_tokens"])
            self.calls+=1;self.tokens+=result.generated_tokens
        if path:write_json(path,{"prompt":text,"prediction":asdict(result)})
        return result


def repair_once(grid,original,method,predict):
    diagnosis=execute(grid,original.text)
    detail={"feedback":diagnosis.verification_label,"model_calls":0}
    if diagnosis.success:return original,{**detail,"status":"valid_preserved"}
    if method=="prompt":
        prompt=render_repair(grid,original.text,diagnosis.verification_label)
        output=predict(prompt)
        return output,{**detail,"status":"prompt_repair","model_calls":1,"prompt":prompt,"output":asdict(output)}
    if method!="prefix":raise ValueError("v1仅允许prompt/prefix")
    # 首错前动作已由执行器实际成功执行；不搜索可行后缀。
    prefix=diagnosis.actions[:diagnosis.executed_steps]
    subgrid=Grid(grid.rows,grid.cols,grid.obstacles,diagnosis.final_position,grid.goal)
    detail.update(prefix=list(prefix),subproblem=grid_to_dict(subgrid))
    if subgrid.start==subgrid.goal:
        return Prediction(" ".join(prefix),0),{**detail,"status":"empty_suffix_at_goal"}
    prompt=render_generation(subgrid);suffix=predict(prompt)
    output=Prediction(" ".join((*prefix,*normalize_actions(suffix.text))),suffix.generated_tokens,suffix.failure)
    return output,{**detail,"status":"prefix_regeneration","model_calls":1,"prompt":prompt,"suffix":asdict(suffix)}


def evaluate_case(row,predict,config):
    grid=grid_from_record(row)
    initial=(predict(render_generation(grid)) if row["task"]=="GENERATE"
             else Prediction(" ".join(row["candidate_plan"]),0))
    original,scoring=score_path(row,initial,config["max_plan_actions"])
    outcomes={}
    for method in ("prompt","prefix"):
        final,detail=repair_once(grid,initial,method,predict)
        scored,final_scoring=score_path(row,final,config["max_plan_actions"])
        outcomes[method]={"prediction":asdict(final),"success":scored.success,"feasible":scored.feasible,
            "detail":detail,"scoring":final_scoring}
    return {"sample_id":row["sample_id"],"initial":asdict(initial),"initial_scoring":scoring,
            "initial_success":original.success,"executor_feedback":execute(grid,initial.text).verification_label,
            "outcomes":outcomes}


def path_report(rows,predictions,config):
    results=[];failures=[]
    for row,p in zip(rows,predictions,strict=True):
        result,details=score_path(row,p,config["max_plan_actions"])
        results.append(result);failures.append(details["interface_failure"])
    distances=[r["shortest_distance"] for r in rows]
    metrics=aggregate_path_metrics(results,distances)
    successful=[(r,d,p) for r,d,p in zip(results,distances,predictions) if r.success]
    physical=[r for r in results if r.first_error_step is not None]
    n=len(rows)
    metrics.update(interface_failure_count=sum(f is not None for f in failures),
        action_token_compliance_rate=sum(all(a in ACTION_ORDER for a in normalize_actions(p.text)) for p in predictions)/n if n else None,
        interface_failures=dict(Counter(f for f in failures if f)),
        mean_output_actions=sum(len(normalize_actions(p.text)) for p in predictions)/n if n else None,
        mean_success_actions=sum(r.executed_steps for r,_,_ in successful)/len(successful) if successful else None,
        mean_success_excess_actions=sum(r.executed_steps-d for r,d,_ in successful)/len(successful) if successful else None,
        first_error_reasons=dict(Counter(r.error_reason.value for r in physical)),
        mean_first_error_step=sum(r.first_error_step for r in physical)/len(physical) if physical else None,
        exact_reference_match_count=sum(p.failure is None and normalize_actions(p.text)==tuple(r["reference_plan"])
                                        for r,p in zip(rows,predictions)))
    return metrics,results


def summarize_panel(rows,cases,config):
    by_id={c["sample_id"]:c for c in cases}
    cases=[by_id[r["sample_id"]] for r in rows]
    initial=[Prediction(**c["initial"]) for c in cases]
    direct,originals=path_report(rows,initial,config)
    result={"count":len(rows),"direct":direct,"repairs":{}}
    if rows and "verification_label" in rows[0]:
        result["executor_verification_consistency"]=aggregate_verification_metrics(
            [r["verification_label"] for r in rows],[c["executor_feedback"] for c in cases])
    for method in ("prompt","prefix"):
        final=[Prediction(**c["outcomes"][method]["prediction"]) for c in cases]
        paths,finals=path_report(rows,final,config)
        stats=aggregate_repair_metrics(originals,finals)
        status=Counter(c["outcomes"][method]["detail"]["status"] for c in cases)
        gain=[c["sample_id"] for c,o,f in zip(cases,originals,finals) if not o.success and f.success]
        loss=[c["sample_id"] for c,o,f in zip(cases,originals,finals) if o.success and not f.success]
        result["repairs"][method]={**paths,**stats,"net_success_gain":len(gain)-len(loss),
            "gained_ids":gain,"lost_ids":loss,"statuses":dict(status),
            "logical_repair_model_calls":sum(c["outcomes"][method]["detail"]["model_calls"] for c in cases),
            "empty_suffix_success_count":sum(c["outcomes"][method]["detail"]["status"]=="empty_suffix_at_goal" and f.success
                                               for c,f in zip(cases,finals)),
            "neural_suffix_success_count":sum(c["outcomes"][method]["detail"]["status"]=="prefix_regeneration"
                                               and not o.success and f.success for c,o,f in zip(cases,originals,finals))}
    result["paired_comparison"]={"prompt_only_success_count":sum(c["outcomes"]["prompt"]["success"] and not c["outcomes"]["prefix"]["success"] for c in cases),
        "prefix_only_success_count":sum(c["outcomes"]["prefix"]["success"] and not c["outcomes"]["prompt"]["success"] for c in cases),
        "both_success_count":sum(c["outcomes"]["prefix"]["success"] and c["outcomes"]["prompt"]["success"] for c in cases)}
    # 分组不递归生成子分组，分母与类别数量一并保存。
    grouped={}
    selectors={"obstacle_count":lambda r,c:str(len(r["obstacles"])),
        "reference_distance":lambda r,c:"2-4" if r["shortest_distance"]<=4 else "5-7" if r["shortest_distance"]<=7 else "8+",
        "original_error":lambda r,c:c["initial_scoring"]["execution"]["error_reason"] or ("valid" if c["initial_success"] else "interface_failure")}
    for kind,select in selectors.items():
        groups={}
        for row,case in zip(rows,cases):groups.setdefault(select(row,case),[]).append((row,case))
        grouped[kind]={}
        for label,items in groups.items():
            rs,cs=zip(*items)
            d,orig=path_report(rs,[Prediction(**c["initial"]) for c in cs],config)
            entry={"count":len(rs),"direct":d,"repairs":{}}
            for method in ("prompt","prefix"):
                p,final=path_report(rs,[Prediction(**c["outcomes"][method]["prediction"]) for c in cs],config)
                entry["repairs"][method]={**p,**aggregate_repair_metrics(orig,final)}
            grouped[kind][label]=entry
    result["by_group"]=grouped
    return result


def evaluate_generation(model,tokenizer,rows,config,budget=None):
    """仅G开发指标选检查点；不借修复指标挑选权重。"""
    from .training_helpers import collate
    from .inference import autocast_context
    import torch
    predictor=Predictor(model,tokenizer,config,budget)
    model.eval();predictions=[predictor(r["input_text"]) for r in rows]
    loss_sum=tokens=0
    for i in range(0,len(rows),config["eval_batch_size"]):
        if budget:budget.check(reserve_seconds=30)
        features=[{"input_ids":tokenizer(r["input_text"])["input_ids"],
                   "labels":tokenizer(r["target_text"])["input_ids"]} for r in rows[i:i+config["eval_batch_size"]]]
        batch=collate(features,tokenizer.pad_token_id,predictor.device)
        count=int((batch["labels"]!=-100).sum())
        with torch.inference_mode(),autocast_context(predictor.device,config["precision"]):
            loss=model(**batch).loss
        if not torch.isfinite(loss):raise FloatingPointError("开发loss非有限")
        loss_sum+=float(loss)*count;tokens+=count
    metrics,_=path_report(rows,predictions,config)
    return {"count":len(rows),"metrics":metrics,"nll":loss_sum/tokens,
        "predictions":[{"sample_id":r["sample_id"],**asdict(p)} for r,p in zip(rows,predictions)],
        "performance":{"model_calls":predictor.calls,"seconds":predictor.seconds,"generated_tokens":predictor.tokens}}
