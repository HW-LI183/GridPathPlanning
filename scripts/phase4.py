#!/usr/bin/env python3
"""阶段四v1：预检、计入预算的pilot/训练、评估与选模。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from gridpath.training_config import ROOT,binding_for,read_json,write_json,model_keys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("preflight","pilot","train","evaluate","select-primary","summarize"))
    p.add_argument("--model",choices=model_keys(),default="t5_base")
    p.add_argument("--split",choices=("dev","test_id","test_ood"),default="dev")
    p.add_argument("--models",nargs="+",choices=model_keys(),help="预检/汇总/选模的候选集合；默认四模型")
    a=p.parse_args()
    keys=tuple(a.models or model_keys())
    if len(set(keys))!=len(keys):p.error("模型列表不能重复")
    if a.mode=="preflight":
        cards={}
        for m in keys:
            cfg,train,dev,binding=binding_for(m,verify_weights=True)
            from transformers import AutoTokenizer
            from gridpath.prompts import render_repair
            from gridpath.data_schema import grid_from_record
            from gridpath.executor import execute
            tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg["model_directory"],local_files_only=True)
            source_lengths=[len(tokenizer(r["input_text"])["input_ids"]) for r in train+dev if r["task"]=="GENERATE"]
            target_lengths=[len(tokenizer(r["target_text"])["input_ids"]) for r in train]
            for row in dev:
                if row["task"]!="GENERATE":
                    grid=grid_from_record(row)
                    feedback=execute(grid,row["candidate_plan"]).verification_label
                    source_lengths.append(len(tokenizer(render_repair(grid,row["candidate_plan"],feedback))["input_ids"]))
            if max(source_lengths)>cfg["max_source_tokens"] or max(target_lengths)>cfg["max_target_tokens"]:
                raise ValueError(f"{m}输入/输出超长；禁止静默截断")
            cards[m]={"max_source_tokens_observed":max(source_lengths),"max_target_tokens_observed":max(target_lengths),"fingerprint":binding["fingerprint"],"train_count":len(train),"dev_count":len(dev),
                      "max_updates":cfg["max_updates"],"status":"exists" if (ROOT/cfg["run_directory"]).exists() else "not_started"}
        ledger_path=ROOT/"manifests/gpu_budget.json"
        ledger=read_json(ledger_path) if ledger_path.exists() else {"entries":[]}
        active=[e["id"] for e in ledger["entries"] if e["status"]=="running"]
        if active:raise RuntimeError(f"有未结束GPU会话:{active}")
        used=sum(e["wall_seconds"] for e in ledger["entries"])/3600
        if used>=20:raise RuntimeError("项目GPU预算已耗尽")
        result={"version":"phase4.v1","models":cards,"gpu_hours_used":used,"gpu_hours_remaining":20-used,"gpu_calls":0}
        write_json(ROOT/"results/v1/preflight.json",result)
    elif a.mode in ("pilot","train"):
        from gridpath.training import run_training
        result=run_training(a.model,pilot=a.mode=="pilot")
    elif a.mode=="evaluate":
        from gridpath.training import run_evaluation
        result=run_evaluation(a.model,a.split)
    elif a.mode=="select-primary":
        from gridpath.training import select_primary
        result=select_primary(models=keys)
    else:
        models={}
        for m in keys:
            path=ROOT/f"results/v1/{m}/{a.split}/summary.json"
            if path.exists():models[m]=read_json(path)
        result={"version":"phase4.v1","split":a.split,"complete":len(models)==len(keys),"expected_models":list(keys),"models":models}
        write_json(ROOT/f"results/v1/summary_{a.split}.json",result)
        import csv,io
        fields=["model","panel","method","count","success_rate","feasibility","optimal_success_rate",
            "conditional_optimal_rate","mean_success_optimality_ratio","repair_success_count","original_invalid_count",
            "repair_success_rate","valid_destruction_rate","mean_success_actions","mean_success_excess_actions",
            "interface_failure_count","logical_repair_model_calls","empty_suffix_success_count","neural_suffix_success_count"]
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
        for model,report in models.items():
            for panel,body in report["panels"].items():
                for method,metrics in (("direct",body["direct"]),*body["repairs"].items()):
                    writer.writerow({"model":model,"panel":panel,"method":method,
                        **{key:metrics.get(key) for key in fields[3:]}})
        (ROOT/f"results/v1/metrics_{a.split}.csv").write_text(buffer.getvalue())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get("status")=="stopped":raise SystemExit(130)


if __name__=="__main__":main()
