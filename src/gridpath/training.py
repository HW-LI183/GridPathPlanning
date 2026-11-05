"""阶段四v1单任务G训练；三遍上限、pilot计入预算、严格续跑。"""
import signal
import time

from .training_config import ROOT,binding_for,read_json,write_json,sha256
from .training_helpers import (seed_everything,stream_indices,train_update,save_checkpoint,
    validate_checkpoint,load_training_state,require_gpu)
from .resources import GPUBudget


def validate_state(state,progress,config,count=4000):
    step=state["global_step"];consumed=state["consumed_examples"]
    if (consumed!=step*config["effective_batch_size"] or not 0<=step<=config["max_updates"]
            or consumed>count*config["max_total_epochs"]
            or state["evaluated_steps"]!=[s for s in config["evaluate_steps"] if s<step]
               and state["evaluated_steps"]!=[s for s in config["evaluate_steps"] if s<=step]):
        raise ValueError("更新数/暴露/三遍评估游标错误")
    if progress!={"reserved_examples":consumed,"completed_updates":step}:
        raise RuntimeError("快照后有未提交暴露；拒绝回放旧样本，请检查故障记录")


def selection_key(report):
    return [report["metrics"]["success_rate"],report["metrics"]["optimal_success_rate"],-report["nll"]]


def run_training(model_key,pilot=False):
    import torch
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    from .evaluation import evaluate_generation
    cfg,train,dev,binding=binding_for(model_key)
    run=ROOT/cfg["run_directory"];run.mkdir(parents=True,exist_ok=True)
    if (run/"run.json").exists() and read_json(run/"run.json")!=binding:
        raise ValueError("配置/源码/数据变化，禁止覆盖v1训练")
    write_json(run/"run.json",binding)
    if (run/"completed.json").exists():
        meta=validate_checkpoint(run/"latest",binding["fingerprint"])
        if meta["state"]["consumed_examples"]!=len(train)*cfg["max_total_epochs"]:raise ValueError("完成记录暴露数不符")
        return read_json(run/"completed.json")
    if pilot and (run/"pilot_report.json").exists() and (run/"latest").exists():
        meta=validate_checkpoint(run/"latest",binding["fingerprint"])
        validate_state(meta["state"],read_json(run/"progress.json"),cfg)
        if meta["state"]["global_step"]>=cfg["pilot_updates"]:
            return read_json(run/"pilot_report.json")
    require_gpu(cfg)
    with GPUBudget(ROOT/"manifests/gpu_budget.json",cfg["gpu_hours_cap"],cfg["session_wall_seconds"],
                   f"v1:{model_key}:{'pilot' if pilot else 'train'}") as budget:
        seed_everything(42)
        tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg["model_directory"],local_files_only=True)
        features=[]
        for row in train:
            x=tokenizer(row["input_text"])["input_ids"];y=tokenizer(row["target_text"])["input_ids"]
            if len(x)>cfg["max_source_tokens"] or len(y)>cfg["max_target_tokens"]:
                raise ValueError("训练超长，禁止截断")
            features.append({"input_ids":x,"labels":y})
        model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/cfg["model_directory"],local_files_only=True).to("cuda")
        no_decay=[];decay=[]
        for name,p in model.named_parameters():
            (no_decay if name.endswith("bias") or "layer_norm" in name else decay).append(p)
        optimizer=torch.optim.AdamW([{"params":decay,"weight_decay":cfg["weight_decay"]},
            {"params":no_decay,"weight_decay":0.}],lr=cfg["learning_rate"])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(step/cfg["warmup_updates"],1.))
        if (run/"latest").exists():
            meta=validate_checkpoint(run/"latest",binding["fingerprint"])
            state=meta["state"]
            validate_state(state,read_json(run/"progress.json"),cfg)
            # 原始模型结构建立后，加载本运行latest权重和完整优化/RNG状态。
            from safetensors.torch import load_model
            load_model(model,str(run/"latest/model.safetensors"),strict=True,device="cuda")
            loaded=load_training_state(run/"latest",optimizer,scheduler,"cuda")
            if loaded!=state:raise ValueError("训练状态与快照元数据不一致")
        else:
            if (run/"progress.json").exists():raise RuntimeError("有暴露记录但无快照，拒绝重新开始")
            state={"global_step":0,"consumed_examples":0,"evaluated_steps":[],"best_score":None,"best_step":None}
            write_json(run/"progress.json",{"reserved_examples":0,"completed_updates":0})
            save_checkpoint(run,["latest"],model,tokenizer,optimizer,scheduler,state,binding)
        if pilot and state["global_step"]>=cfg["pilot_updates"] and (run/"pilot_report.json").exists():
            return read_json(run/"pilot_report.json")
        if pilot and state["global_step"]>cfg["pilot_updates"]:
            raise RuntimeError("已进入正式训练但缺少pilot报告，请检查故障记录")
        if not pilot and (run/"pilot").is_symlink():(run/"pilot").unlink()
        gdev=[r for r in dev if r["task"]=="GENERATE"]
        probe=[]
        for count in range(1,6):
            probe.extend(sorted((r for r in train if len(r["obstacles"])==count),key=lambda r:r["sample_id"])[:20])
        def checkpoint(aliases):
            save_checkpoint(run,aliases,model,tokenizer,optimizer,scheduler,state,binding)
        def evaluate_pending():
            step=state["global_step"]
            if step not in cfg["evaluate_steps"] or step in state["evaluated_steps"]:return
            report=evaluate_generation(model,tokenizer,gdev,cfg,budget)
            seen=evaluate_generation(model,tokenizer,probe,cfg,budget)
            score=selection_key(report);best=state["best_score"] is None or score>state["best_score"]
            state["evaluated_steps"].append(step)
            if best:state.update(best_score=score,best_step=step)
            write_json(run/f"evaluations/step_{step:04d}.json",{"fingerprint":binding["fingerprint"],
                "step":step,"dev_generate":report,"train_seen100":seen,"selection":"dev_generate100"})
            checkpoint(["latest","best"] if best else ["latest"])
        stop={"requested":False}
        def request_stop(*_):stop["requested"]=True
        previous={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
        began=time.perf_counter()
        target=cfg["pilot_updates"] if pilot else cfg["max_updates"]
        try:
            evaluate_pending()
            while state["global_step"]<target:
                if stop["requested"] or budget.remaining_seconds<60:
                    checkpoint(["latest"])
                    return {"status":"stopped","state":state}
                indices=stream_indices(len(features),42,state["consumed_examples"],cfg["effective_batch_size"])
                write_json(run/"progress.json",{"reserved_examples":state["consumed_examples"]+len(indices),
                                               "completed_updates":state["global_step"]})
                update=train_update(model,optimizer,scheduler,[features[i] for i in indices],tokenizer,cfg)
                state["global_step"]+=1;state["consumed_examples"]+=len(indices)
                write_json(run/"progress.json",{"reserved_examples":state["consumed_examples"],"completed_updates":state["global_step"]})
                import json
                with (run/"training.jsonl").open("a") as f:
                    f.write(json.dumps({"step":state["global_step"],"consumed":state["consumed_examples"],**update})+"\n")
                if state["global_step"]%cfg["checkpoint_every_updates"]==0:checkpoint(["latest"])
                evaluate_pending()
                if state["global_step"]%100==0:print(f"{model_key}: {state['global_step']}/3000",flush=True)
            if pilot:
                report=evaluate_generation(model,tokenizer,gdev[:4],cfg,budget)
                checkpoint(["latest","pilot"])
                result={"status":"pilot_complete","state":state,"dev4":report,
                    "seconds":time.perf_counter()-began,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                    "note":"10更新/40样本暴露包含在三遍中；dev4不参与选择"}
                write_json(run/"pilot_report.json",result)
            else:
                validate_state(state,read_json(run/"progress.json"),cfg)
                checkpoint(["latest"])
                result={"status":"complete","state":state,"fingerprint":binding["fingerprint"],
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated()}
                write_json(run/"completed.json",result)
            return result
        except BaseException:
            # 安全评估中断可继续评估；参数更新期间异常由progress不一致阻止回放。
            if read_json(run/"progress.json")=={"reserved_examples":state["consumed_examples"],"completed_updates":state["global_step"]}:
                checkpoint(["latest"])
            raise
        finally:
            for s,handler in previous.items():signal.signal(s,handler)


def run_evaluation(model_key,split="dev"):
    import torch
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    from .evaluation import Predictor,evaluate_case,summarize_panel
    from .training_config import read_rows,digest
    cfg,train,dev,binding=binding_for(model_key,verify_weights=False)
    run=ROOT/cfg["run_directory"]
    completed=read_json(run/"completed.json")
    if completed["fingerprint"]!=binding["fingerprint"] or completed["state"]["consumed_examples"]!=12000:
        raise ValueError("只评估当前v1完整三遍的模型")
    meta=validate_checkpoint(run/"best",binding["fingerprint"])
    manifest=read_json(ROOT/"manifests/dataset_manifest.json")
    if split=="dev":rows=dev
    elif split in ("test_id","test_ood"):
        lock=read_json(ROOT/"configs/primary_model_v1.json")
        if lock["data_manifest_sha256"]!=sha256(ROOT/"manifests/dataset_manifest.json"):
            raise ValueError("最终测试需匹配当前开发选择锁")
        current=select_primary(write=False)
        if lock!=current:raise ValueError("最终测试的主模型锁已失效")
        # 每个既有测试episode既生成一次，也评价它自己的受控候选；不新增环境。
        controlled=read_rows(ROOT/manifest["files"][split]["path"])
        generated=[{**{k:v for k,v in r.items() if k not in
            ("candidate_plan","base_plan","edit_operations","is_valid","verification_label","error_reason")},
            "task":"GENERATE","sample_id":r["sample_id"]+"_generate"} for r in controlled]
        rows=generated+controlled
    else:raise ValueError("划分只能dev/test_id/test_ood")
    directory=ROOT/f"results/v1/{model_key}/{split}";directory.mkdir(parents=True,exist_ok=True)
    if split=="dev" and (ROOT/"manifests/v1_training_lineage.json").exists():
        entry=read_json(ROOT/"manifests/v1_training_lineage.json")["models"][model_key]
        for sample_id,expected in entry["retained_generation_cases"].items():
            path=directory/"cases"/f"{sample_id}.json"
            if not path.exists() or sha256(path)!=expected:
                raise ValueError("冻结G100结果缺失或改变；本轮仅允许重新评价R100")
    evaluation_binding={"training_fingerprint":binding["fingerprint"],"best_files":meta["files"],"split":split,
        "dataset_manifest_sha256":sha256(ROOT/"manifests/dataset_manifest.json"),
        "repair_dev_revision":"invalid100"}
    fingerprint=digest(evaluation_binding)
    if (directory/"binding.json").exists() and read_json(directory/"binding.json")!=evaluation_binding:
        raise ValueError("评估绑定不同，不能混用结果")
    write_json(directory/"binding.json",evaluation_binding)
    if (directory/"summary.json").exists():
        result=read_json(directory/"summary.json")
        if result["fingerprint"]!=fingerprint or result["count"]!=len(rows):raise ValueError("已有汇总不符")
        return result
    require_gpu(cfg)
    with GPUBudget(ROOT/"manifests/gpu_budget.json",20,cfg["session_wall_seconds"],f"v1:{model_key}:eval:{split}") as budget:
        seed_everything(42)
        tokenizer=AutoTokenizer.from_pretrained(run/"best",local_files_only=True)
        model=AutoModelForSeq2SeqLM.from_pretrained(run/"best",local_files_only=True).to("cuda").eval()
        predictor=Predictor(model,tokenizer,cfg,budget,directory/"cache")
        cases=[];began=time.perf_counter()
        for row in rows:
            budget.check(reserve_seconds=35)
            path=directory/"cases"/f"{row['sample_id']}.json"
            case=read_json(path) if path.exists() else evaluate_case(row,predictor,cfg)
            if case["sample_id"]!=row["sample_id"]:raise ValueError("案例缓存不符")
            write_json(path,case);cases.append(case)
            print(f"{model_key}/{split}: {len(cases)}/{len(rows)}",flush=True)
        generated=[r for r in rows if r["task"]=="GENERATE"]
        repair=[r for r in rows if r["task"]!="GENERATE"]
        best_report=read_json(run/f"evaluations/step_{meta['state']['best_step']:04d}.json")
        curve=[]
        for step in cfg["evaluate_steps"]:
            item=read_json(run/f"evaluations/step_{step:04d}.json")
            curve.append({"step":step,"epoch":step//1000,
                "dev_generate_metrics":item["dev_generate"]["metrics"],"dev_generate_nll":item["dev_generate"]["nll"],
                "train_seen100_metrics":item["train_seen100"]["metrics"],"train_seen100_nll":item["train_seen100"]["nll"]})
        result={"version":"phase4.v1","fingerprint":fingerprint,"count":len(rows),
            "best_step":meta["state"]["best_step"],"panels":{"generated":summarize_panel(generated,cases,cfg),
              "controlled_repair":summarize_panel(repair,cases,cfg)},
            "training_curve":curve,"dev_generate_selection_nll":best_report["dev_generate"]["nll"],
            "performance":{"new_model_calls":predictor.calls,"cache_hits":predictor.hits,
                "generation_seconds":predictor.seconds,"generated_tokens":predictor.tokens,
                "generated_tokens_per_second":predictor.tokens/predictor.seconds if predictor.seconds else None,
                "evaluation_seconds":time.perf_counter()-began,"peak_allocated_bytes":torch.cuda.max_memory_allocated()},
            "cost_note":"逻辑修复调用按组统计；同提示缓存共享。GPU累计成本看全局账本，非本会话秒数。"}
        if split=="dev":
            current=result["panels"]["generated"]["direct"]
            expected=best_report["dev_generate"]["metrics"]
            if current["success_count"]!=expected["success_count"] or current["optimal_success_count"]!=expected["optimal_success_count"]:
                raise ValueError("最终G100参照没有复现选中检查点的开发成绩")
        write_json(directory/"summary.json",result)
        return result


def select_primary(write=True):
    cards=[]
    for key in ("t5_base","t5_small"):
        cfg,_,_,binding=binding_for(key,verify_weights=False)
        run=ROOT/cfg["run_directory"];complete=read_json(run/"completed.json")
        if complete["fingerprint"]!=binding["fingerprint"] or complete["state"]["consumed_examples"]!=12000:
            raise ValueError("两个模型须完成三遍")
        meta=validate_checkpoint(run/"best",binding["fingerprint"])
        step=meta["state"]["best_step"]
        if step!=complete["state"]["best_step"]:raise ValueError("最佳快照与完成状态不一致")
        report=read_json(run/f"evaluations/step_{step:04d}.json")
        if report["fingerprint"]!=binding["fingerprint"] or report["dev_generate"]["count"]!=100:
            raise ValueError("选模只接受当前G100完整报告")
        cards.append({"model":key,"step":step,"fingerprint":binding["fingerprint"],
            "model_sha256":meta["files"]["model.safetensors"],"metrics":report["dev_generate"]["metrics"]})
    selected=max(cards,key=lambda c:(c["metrics"]["success_rate"],c["metrics"]["optimal_success_rate"],c["model"]=="t5_base"))
    result={"version":"phase4.v1","selection":"仅dev_generate100；成功率、最优成功率，平局固定base",
        "data_manifest_sha256":sha256(ROOT/"manifests/dataset_manifest.json"),"selected":selected,"candidates":cards}
    path=ROOT/"configs/primary_model_v1.json"
    if write:
        if path.exists() and read_json(path)!=result:raise ValueError("主模型锁已有不同内容")
        write_json(path,result)
    return result
