"""阶段三v1：4000G、开发G100/R100；CPU确定性构造。"""
import random
import time
from collections import Counter

import yaml

from .bfs import shortest_path
from .corruptions import enumerate_single_corruptions, edit_to_dict
from .data_schema import canonical_hash, environment_id, grid_from_record, grid_to_dict, layout_id, layout_split
from .executor import execute
from .prompts import render_generation
from .training_config import ROOT, read_json, sha256, write_json
from .types import ErrorReason, Grid

FILES = ("train", "dev", "test_id", "test_ood", "test_environments")
DATA_SOURCES = ("configs/data.yaml", "configs/project.json", "data/v1_sources.json",
    "src/gridpath/data.py", "src/gridpath/data_audit.py", "src/gridpath/prompts.py",
    "src/gridpath/data_schema.py", "src/gridpath/corruptions.py", "src/gridpath/bfs.py",
    "src/gridpath/executor.py", "src/gridpath/types.py")


def record(grid, split, task, candidate=None, base=None, edits=None):
    reference = shortest_path(grid)
    if not reference.reachable or not 2 <= reference.distance <= 72:
        raise ValueError("数据环境需可达，参考距离至少2")
    row = {"version":"phase3.v1", "split":split, "task":task, **grid_to_dict(grid),
        "layout_id":layout_id(grid), "environment_id":environment_id(grid),
        "reference_plan":list(reference.actions), "shortest_distance":reference.distance,
        "input_text":render_generation(grid), "target_text":" ".join(reference.actions)}
    if candidate is not None:
        result = execute(grid, candidate)
        row.update(candidate_plan=list(candidate), base_plan=list(base or reference.actions),
            edit_operations=edits or [], is_valid=result.success,
            verification_label=result.verification_label,
            error_reason=result.error_reason.value if result.error_reason else None)
    row["sample_id"] = canonical_hash({"version":"phase3.v1", "split":split, "task":task,
        "environment":row["environment_id"], "candidate":row.get("candidate_plan")}, "sample")
    return row


def dev_rows(config, used):
    rng = random.Random(config["seed"])
    cells = [(r,c) for r in range(6) for c in range(6)]
    def environment(obstacles):
        for _ in range(20000):
            blocked = frozenset(rng.sample(cells, obstacles))
            free = [p for p in cells if p not in blocked]
            start, goal = rng.sample(free, 2)
            grid = Grid(6,6,blocked,start,goal)
            if layout_split(layout_id(grid),42) != "dev" or environment_id(grid) in used:
                continue
            reference = shortest_path(grid)
            if reference.reachable and reference.distance >= 2:
                used.add(environment_id(grid))
                return grid, reference
        raise RuntimeError("开发环境有界采样超限")
    # G100顺序与内容独立冻结；只修改修复配额时不得影响检查点选择。
    source=read_json(ROOT/config["sources"])
    generated=[record(grid_from_record(g),"dev","GENERATE")
               for g in source["dev_generation_environments"]]
    if len(generated)!=config["dev_generate"]:raise ValueError("冻结G100数量错误")
    used.update(r["environment_id"] for r in generated)
    q = config["dev_repair"]
    slots = [reason for reason in ErrorReason for _ in range(q["invalid_per_reason"])]
    repaired = []
    for i, slot in enumerate(slots):
        for _ in range(300):
            grid, reference = environment(i % 5 + 1)
            choices = [c for c in enumerate_single_corruptions(grid, reference.actions)
                if execute(grid,c.actions).error_reason == slot]
            if not choices:continue
            corruption = rng.choice(choices)
            candidate = tuple("jump" if x == "<invalid>" else x for x in corruption.actions)
            edits = [edit_to_dict(e) | ({"token":"jump"} if e.token == "<invalid>" else {})
                     for e in corruption.edits]
            repaired.append(record(grid,"dev","REPAIR",candidate,
                base=reference.actions, edits=edits))
            break
        else:
            raise RuntimeError("开发受控错误配额采样超限")
    rng.shuffle(repaired)
    return generated + repaired


def build_dataset():
    began = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    if (cfg["version"] != "phase3.v1" or cfg["train_generate"] != 4000 or cfg["dev_generate"] != 100
            or cfg["dev_repair"] != {"invalid_per_reason":25}):
        raise ValueError("v1固定4000G、开发100G/100R")
    source = read_json(ROOT / cfg["sources"])
    train = [record(grid_from_record(g),"train","GENERATE") for g in source["train_environments"]]
    used = {r["environment_id"] for r in train}
    if len(train) != 4000 or len(used) != 4000:
        raise ValueError("4000个训练环境必须唯一")
    dev = dev_rows(cfg, used)
    groups = {"train":train,"dev":dev,"test_id":[],"test_ood":[],"test_environments":[]}
    for episode in source["test_episodes"]:
        grid = grid_from_record(episode["grid"])
        for role, split in (("id","test_id"),("ood","test_ood")):
            case = episode[role]
            groups[split].append(record(grid,split,"EVALUATE",case["candidate_plan"],
                base=case["base_plan"],edits=case["edit_operations"]))
        groups["test_environments"].append(record(grid,"test","GENERATE"))
    if {k:len(v) for k,v in groups.items()} != {"train":4000,"dev":200,"test_id":200,"test_ood":200,"test_environments":200}:
        raise ValueError("数据配额不符")
    directory = ROOT / "data/v1"
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    import json
    for name, rows in groups.items():
        path = directory / f"{name}.jsonl"
        content = "".join(json.dumps(r, ensure_ascii=False,sort_keys=True)+"\n" for r in rows)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content)
        if path.exists() and path.read_text() != content:
            temporary.unlink()
            raise ValueError("已有v1数据不同，拒绝隐式覆盖")
        temporary.replace(path)
        files[name] = {"path":str(path.relative_to(ROOT)),"count":len(rows),"sha256":sha256(path)}
    manifest = {"version":"phase3.v1","files":files,
        "sources":{p:sha256(ROOT/p) for p in DATA_SOURCES},
        "train_tasks":{"GENERATE":4000,"VERIFY":0,"REPAIR":0},
        "dev_tasks":{"GENERATE":100,"VERIFY":0,"REPAIR":100},
        "train_obstacle_counts":dict(Counter(str(len(r["obstacles"])) for r in train)),
        "build_cpu_seconds":time.perf_counter()-began,
        "test_note":"ID/OOD共用200环境；test_environments是索引，不增加测试案例数"}
    existing = ROOT / "manifests/dataset_manifest.json"
    if existing.exists() and read_json(existing).get("version") == "phase3.v1":
        previous = read_json(existing)
        if previous["files"] == files and previous["sources"] == manifest["sources"]:
            return previous
        raise ValueError("v1清单已冻结，需明确登记新版本")
    write_json(existing,manifest)
    return manifest
