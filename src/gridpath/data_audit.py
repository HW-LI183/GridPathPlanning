"""阶段三v1独立审计：布局归属、BFS、受控编辑、官方执行器与token长度。"""
from collections import Counter
import time

from .bfs import shortest_path
from .corruptions import apply_edits,edit_from_dict
from .data_schema import environment_id,grid_from_record,layout_id,layout_split
from .executor import execute
from .official import load_official_is_goal
from .prompts import render_generation,render_repair
from .training_config import ROOT,read_json,read_rows,sha256,write_json
from .types import ACTION_ORDER,ErrorReason


def audit_dataset():
    began = time.perf_counter()
    path = ROOT / "manifests/dataset_manifest.json"
    manifest = read_json(path)
    probe = read_json(ROOT/"manifests/upstream_probe.json")
    official = load_official_is_goal(ROOT/probe["source"],expected_sha256=probe["source_sha256"])
    errors, groups, layouts = [], {}, {}
    expected_counts={"train":4000,"dev":200,"test_id":200,"test_ood":200,"test_environments":200}
    if manifest["version"]!="phase3.v1" or set(manifest["files"])!=set(expected_counts):
        raise ValueError("不是完整阶段三v1清单")
    compared = 0
    for p,h in manifest["sources"].items():
        if sha256(ROOT/p) != h: errors.append(f"来源哈希不符:{p}")
    for name,item in manifest["files"].items():
        file = ROOT/item["path"]
        if sha256(file) != item["sha256"]: errors.append(f"文件哈希不符:{name}")
        rows = read_rows(file);groups[name]=rows
        if len(rows)!=expected_counts[name]:errors.append(f"正式配额不符:{name}")
        if len(rows)!=item["count"] or len({r["sample_id"] for r in rows})!=len(rows):
            errors.append(f"数量/样本ID重复:{name}")
        layouts[name] = set()
        for row in rows:
            grid=grid_from_record(row); ref=shortest_path(grid)
            owner = "test" if name.startswith("test") else name
            layouts[name].add(layout_id(grid))
            if layout_split(layout_id(grid),42)!=owner: errors.append(f"布局归属:{row['sample_id']}")
            if (environment_id(grid)!=row["environment_id"] or layout_id(grid)!=row["layout_id"]
                    or ref.distance!=row["shortest_distance"] or list(ref.actions)!=row["reference_plan"]
                    or row["input_text"]!=render_generation(grid)):
                errors.append(f"参考/身份/提示错误:{row['sample_id']}")
            if not execute(grid,row["target_text"]).success: errors.append("监督路径失败")
            candidate=row.get("candidate_plan",ref.actions);result=execute(grid,candidate)
            if "verification_label" in row and result.verification_label!=row["verification_label"]:
                errors.append("候选真值不符")
            edits=[edit_from_dict(e) for e in row.get("edit_operations",[])]
            if "candidate_plan" in row and not result.success and not edits:errors.append("无效候选缺少受控编辑证据")
            if edits:
                if not execute(grid,row["base_plan"]).success:errors.append("编辑父路径本身无效")
                if list(apply_edits(row["base_plan"],edits))!=list(candidate): errors.append("编辑无法重放")
                if result.success: errors.append("受控错误仍成功")
                if name=="test_ood" and (len(edits)!=2 or any(execute(grid,apply_edits(row["base_plan"],[e])).success for e in edits)):
                    errors.append("OOD双错反事实不符")
                if name in ("dev","test_id") and len(edits)!=1: errors.append("单错编辑数不符")
            if result.actions and all(a in ACTION_ORDER for a in result.actions):
                expected=1 if result.success else 2 if result.error_reason==ErrorReason.OBSTACLE_COLLISION else 0
                if official(grid.to_world(),grid.start," ".join(result.actions),grid.rows,grid.cols)!=expected:
                    errors.append("官方执行器不一致")
                compared+=1
    if layouts["train"] & layouts["dev"] or layouts["train"] & layouts["test_id"] or layouts["dev"] & layouts["test_id"]:
        errors.append("train/dev/test布局交叉")
    if Counter(r["task"] for r in groups["train"])!={"GENERATE":4000}: errors.append("训练非4000G")
    if Counter(r["task"] for r in groups["dev"])!={"GENERATE":100,"REPAIR":100}: errors.append("开发非100G/100R")
    repair=[r for r in groups["dev"] if r["task"]=="REPAIR"]
    if Counter(r["error_reason"] or "valid" for r in repair)!={r.value:25 for r in ErrorReason}:
        errors.append("开发修复类别不平衡")
    for name in ("train","dev"):
        if len({r["environment_id"] for r in groups[name]})!=len(groups[name]):errors.append("训练/开发环境重复")
    if {r["environment_id"] for r in groups["test_id"]}!={r["environment_id"] for r in groups["test_ood"]}:
        errors.append("ID/OOD环境未配对")
    for name in ("test_id","test_ood"):
        if Counter(r["error_reason"] or "valid" for r in groups[name])!={"valid":100,**{r.value:25 for r in ErrorReason}}:
            errors.append(f"测试有效/错误类型配额不符:{name}")
    from transformers import AutoTokenizer
    token_lengths={}
    for model in ("t5-small","t5-base"):
        tok=AutoTokenizer.from_pretrained(ROOT/"models"/model,local_files_only=True)
        stats={}
        for name,rows in groups.items():
            inputs=[r["input_text"] for r in rows]
            inputs += [render_repair(grid_from_record(r),r["candidate_plan"],r["verification_label"]) for r in rows if "candidate_plan" in r]
            source=tok(inputs,truncation=False)["input_ids"]
            targets=tok([r["target_text"] for r in rows],truncation=False)["input_ids"]
            if any(len(x)>512 or tok.unk_token_id in x for x in source) or any(len(x)>256 for x in targets):
                errors.append(f"token超限/未知词:{model}/{name}")
            stats[name]={"max_source":max(map(len,source)),"max_target":max(map(len,targets))}
        # 72动作运行期修复上下文的保守检查。
        row=max(groups["dev"],key=lambda r:len(r["obstacles"]))
        worst=max(len(tok(render_repair(grid_from_record(row),[a]*72,"invalid | step=73 | reason=obstacle_collision"))["input_ids"]) for a in ACTION_ORDER)
        stats["repair_72_action_source_tokens"]=worst
        if worst>512: errors.append("72动作修复上下文超限")
        token_lengths[model]=stats
    report={"version":"phase3.v1","status":"passed" if not errors else "failed","errors":errors,
        "manifest_sha256":sha256(path),"counts":{k:len(v) for k,v in groups.items()},
        "layout_counts":{k:len(v) for k,v in layouts.items()},"official_comparisons":compared,
        "token_lengths":token_lengths,"cpu_seconds":time.perf_counter()-began}
    write_json(ROOT/"results/data_audit.json",report)
    write_json(ROOT/"manifests/phase3_checks.json",{"version":"phase3.v1","status":report["status"],
        "dataset_manifest_sha256":sha256(path),"audit_report_sha256":sha256(ROOT/"results/data_audit.json")})
    return report
