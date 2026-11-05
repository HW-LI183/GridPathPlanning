#!/usr/bin/env python3
"""短CPU复核v1既有结果并绘图；不训练、不执行真实模型推理。"""
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from gridpath.training_config import ROOT,binding_for,read_json,write_json,read_rows,digest,sha256
from gridpath.training_helpers import validate_checkpoint
from gridpath.training import select_primary
from gridpath.evaluation import summarize_panel
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.inference import Prediction,score_path


def main():
    began=time.perf_counter()
    total=read_json(ROOT/"results/v1/summary_dev.json")
    assert total["complete"] and total["split"]=="dev"
    from gridpath.training_config import primary_lock_path
    models=tuple(total["models"])
    assert total["complete"] and len(models) in (2,4)
    assert read_json(primary_lock_path(models))==select_primary(write=False,models=tuple(total["models"]))
    review={"version":"phase4.v1","gpu_calls":0,"models":{}}
    case_examples={}
    for model in models:
        cfg,train,dev,binding=binding_for(model,verify_weights=False)
        run=ROOT/cfg["run_directory"]
        complete=read_json(run/"completed.json")
        latest=validate_checkpoint(run/"latest",binding["fingerprint"])
        best=validate_checkpoint(run/"best",binding["fingerprint"])
        assert complete["state"]==latest["state"]
        assert complete["state"]["global_step"]==3000 and complete["state"]["consumed_examples"]==12000
        assert complete["state"]["evaluated_steps"]==[1000,2000,3000]
        updates=read_rows(run/"training.jsonl")
        assert len(updates)==3000 and [u["step"] for u in updates]==list(range(1,3001))
        assert all(u["consumed"]==4*u["step"] for u in updates)
        directory=ROOT/f"results/v1/{model}/dev"
        report=read_json(directory/"summary.json")
        assert report==total["models"][model]
        eb={"training_fingerprint":binding["fingerprint"],"best_files":best["files"],"split":"dev",
            "dataset_manifest_sha256":sha256(ROOT/"manifests/dataset_manifest.json"),"repair_dev_revision":"invalid100"}
        assert read_json(directory/"binding.json")==eb and report["fingerprint"]==digest(eb)
        assert len(list((directory/"cases").glob('*.json')))==200
        cases=[read_json(directory/"cases"/f"{r['sample_id']}.json") for r in dev]
        by_id={r["sample_id"]:r for r in dev}
        for row,case in zip(dev,cases,strict=True):
            original=Prediction(**case["initial"])
            original_result,original_scoring=score_path(row,original,cfg["max_plan_actions"])
            assert original_result.success==case["initial_success"]
            assert digest(original_scoring)==digest(case["initial_scoring"])
            truth=execute(grid_from_record(row),original.text)
            assert truth.verification_label==case["executor_feedback"]
            for outcome in case["outcomes"].values():
                final,scoring=score_path(row,Prediction(**outcome["prediction"]),cfg["max_plan_actions"])
                assert final.success==outcome["success"] and final.feasible==outcome["feasible"]
                assert digest(scoring)==digest(outcome["scoring"])
                assert outcome["detail"]["feedback"]==truth.verification_label
                if truth.success:assert outcome["prediction"]==case["initial"]
        for panel,task in (("generated","GENERATE"),("controlled_repair","REPAIR")):
            rows=[r for r in dev if r["task"]==task]
            assert summarize_panel(rows,cases,cfg)==report["panels"][panel]
        best_epoch=read_json(run/f"evaluations/step_{best['state']['best_step']:04d}.json")
        for p in best_epoch["dev_generate"]["predictions"]:
            initial=next(c["initial"] for c in cases if c["sample_id"]==p["sample_id"])
            assert initial=={k:v for k,v in p.items() if k!="sample_id"}
        criteria={
            "base_prefix_collision_success":lambda c: model=="t5_base" and by_id[c["sample_id"]]["task"]=="GENERATE" and
                "obstacle_collision" in c["executor_feedback"] and c["outcomes"]["prefix"]["success"]
                and c["outcomes"]["prefix"]["detail"]["status"]=="prefix_regeneration",
            "base_collision_both_fail":lambda c:model=="t5_base" and by_id[c["sample_id"]]["task"]=="GENERATE" and
                "obstacle_collision" in c["executor_feedback"] and not any(o["success"] for o in c["outcomes"].values()),
            "small_empty_suffix_success":lambda c:model=="t5_small" and by_id[c["sample_id"]]["task"]=="GENERATE" and
                c["outcomes"]["prefix"]["detail"]["status"]=="empty_suffix_at_goal"}
        for name,predicate in criteria.items():
            chosen=next((c for c in sorted(cases,key=lambda c:c["sample_id"]) if predicate(c)),None)
            if chosen:case_examples[name]={"model":model,"row":by_id[chosen["sample_id"]],"case":chosen}
        review["models"][model]={"status":"passed","count":len(cases),"training_updates":len(updates),
            "consumed_examples":12000,"fingerprint":binding["fingerprint"],"best_step":best["state"]["best_step"],
            "metrics_recomputed":True,"checkpoint_reference_reproduced":True,"candidate_traces_replayed":True}
    records=list(csv.DictReader((ROOT/"results/v1/metrics_dev.csv").open()))
    assert len(records)==6*len(models)
    for record in records:
        body=total["models"][record["model"]]["panels"][record["panel"]]
        metrics=body["direct"] if record["method"]=="direct" else body["repairs"][record["method"]]
        for key,value in record.items():
            if key in ("model","panel","method"):continue
            assert value=="" if metrics.get(key) is None else float(value)==metrics[key]
    ledger=read_json(ROOT/"manifests/gpu_budget.json")
    expected_labels={f"v1:{model}:{role}" for model in models
                     for role in ("pilot","train","eval:dev")}
    entries=[e for e in ledger["entries"] if e["label"] in expected_labels]
    assert expected_labels<={e["label"] for e in entries if e["status"]=="complete"}
    review.update(csv_matches=True,primary_selection_matches=True,case_examples=case_examples,
        gpu_entries=entries,gpu_seconds_v1=sum(e["wall_seconds"] for e in entries),
        gpu_hours_total=sum(e["wall_seconds"] for e in ledger["entries"])/3600,
        test_results_present=any((ROOT/f"results/v1/summary_{s}.json").exists() for s in ("test_id","test_ood")))
    review["cpu_review_seconds"]=time.perf_counter()-began
    write_json(ROOT/"results/v1/review_dev.json",review)
    plot(total)
    receipt=read_json(ROOT/"manifests/phase4_checks.json")
    receipt.update(status="dev_completed_test_pending",results_review_sha256=sha256(ROOT/"results/v1/review_dev.json"),
        gpu_hours_remaining=20-review["gpu_hours_total"],user_training_completed=True)
    for model in models:
        receipt["models"].setdefault(model,{}).update(status="complete",best_step=review["models"][model]["best_step"])
    receipt["assistant_formal_training_runs"]=receipt.pop("new_formal_training_runs",0)
    receipt["assistant_real_model_gpu_inference_calls"]=receipt.pop("new_real_model_gpu_inference_calls",0)
    receipt["user_formal_training_model_count"]=len(models)
    if len(models)==4:
        receipt.setdefault("baseline_extension",{}).update(status="four_models_dev_completed",reviewed_case_count=800)
    write_json(ROOT/"manifests/phase4_checks.json",receipt)
    print(json.dumps({"status":"passed","case_count":200*len(models),"csv_matches":True,
        "gpu_hours_total":review["gpu_hours_total"],"cpu_review_seconds":review["cpu_review_seconds"]},ensure_ascii=False))


def plot(total):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    colors=['#2463a6','#e89c35','#23966f','#9b59b6'];models=tuple(total['models'])
    width=.8/len(models)
    for i,m in enumerate(models):
        r=total['models'][m]
        axes[0].plot([1,2,3],[e['dev_generate_metrics']['success_rate']*100 for e in r['training_curve']],
            marker='o',label=m,color=colors[i])
        body=r['panels']['generated']
        vals=[body['direct']['success_rate']*100,*[body['repairs'][k]['success_rate']*100 for k in ('prompt','prefix')]]
        bars=axes[1].bar([x+(i-(len(models)-1)/2)*width for x in range(3)],vals,width=width*.95,color=colors[i],label=m)
        axes[1].bar_label(bars,fmt='%.0f%%',padding=3)
        vals=[r['panels']['controlled_repair']['repairs'][k]['repair_success_rate']*100 for k in ('prompt','prefix')]
        bars=axes[2].bar([x+(i-(len(models)-1)/2)*width for x in range(2)],vals,width=width*.95,color=colors[i],label=m)
        axes[2].bar_label(bars,fmt='%.2f%%',padding=3)
    axes[0].set(title='训练中的G100开发成功率',xlabel='训练遍数',ylabel='成功率（%）',xticks=[1,2,3])
    axes[1].set(title='G100真实生成端到端成功率',xticks=[0,1,2],xticklabels=['直接生成','提示修复','前缀续写'])
    axes[2].set(title='R100全部无效候选的修复率',xticks=[0,1],xticklabels=['提示修复','前缀续写'])
    for ax in axes:ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle('阶段四v1：开发集结果（单种子，非最终测试）',fontsize=13)
    destination=ROOT/'results/v1/figures';destination.mkdir(exist_ok=True)
    fig.savefig(destination/'dev_overview.png',dpi=180)
    fig.savefig(destination/'dev_overview.pdf')
    plt.close(fig)


if __name__=='__main__':main()
