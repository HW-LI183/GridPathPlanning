"""v1统一IO、配置与阶段三冻结检查；训练只解析train/dev。"""
import hashlib
import json
from pathlib import Path
import yaml

from .baselines.layout import WorkspacePath,resolve_artifact,original_source_sha
ROOT = WorkspacePath(Path(__file__).resolve().parents[2])


def sha256(path):
    h = hashlib.sha256()
    with resolve_artifact(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(resolve_artifact(path).read_text(encoding="utf-8"))


def write_json(path, value):
    import os
    path = resolve_artifact(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in resolve_artifact(path).read_text().splitlines() if line.strip()]


def model_keys():
    return ("t5_base", "t5_small", "flan_t5_base", "flan_t5_small")


def primary_lock_path(models=None):
    keys = tuple(models or model_keys())
    if keys == ("t5_base", "t5_small"):
        return ROOT / "configs/primary_model_v1.json"
    if keys != model_keys():
        raise ValueError("主模型选择必须完整比较原T5双模型或全部四模型")
    return ROOT / "configs/primary_model_v1_four.json"


def config_for(model):
    p = yaml.safe_load((ROOT / "configs/phase4_v1.yaml").read_text())
    extra = yaml.safe_load((ROOT / "configs/baselines_extra.yaml").read_text())["models"]
    p["models"] = {**p["models"], **extra}
    if (p["version"] != "phase4.v1" or model not in p["models"]
            or p["max_total_epochs"] != 3 or p["effective_batch_size"] != 4
            or p["micro_batch_size"] != 4 or p["seed"] != 42):
        raise ValueError("v1仅允许已注册的四个T5系列模型、三遍、有效batch4、seed42")
    return {**{k:v for k,v in p.items() if k != "models"}, **p["models"][model],
            "model_key": model, "run_directory": f"checkpoints/v1/{model}",
            "max_updates": 3000, "evaluate_steps": [1000, 2000, 3000]}


def frozen_data():
    manifest = read_json(ROOT / "manifests/dataset_manifest.json")
    audit = read_json(ROOT / "results/data_audit.json")
    if (manifest["version"] != "phase3.v1" or audit["status"] != "passed"
            or audit["manifest_sha256"] != sha256(ROOT / "manifests/dataset_manifest.json")):
        raise ValueError("需要已通过独立审计的阶段三v1")
    for path, expected in manifest["sources"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"阶段三冻结来源变化：{path}")
    for name, item in manifest["files"].items():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"冻结数据变化：{name}")
    train, dev = (read_rows(ROOT / manifest["files"][s]["path"]) for s in ("train", "dev"))
    if len(train) != 4000 or len(dev) != 200 or any(r["task"] != "GENERATE" for r in train):
        raise ValueError("v1训练必须4000G，开发必须200条")
    return train, dev


def binding_for(model, verify_weights=True):
    import importlib.metadata
    cfg = config_for(model)
    train, dev = frozen_data()
    manifest_name = "flan_models.json" if model.startswith("flan_") else "models.json"
    card = next(m for m in read_json(ROOT / "manifests" / manifest_name)["models"] if m["repo_id"] == cfg["model_id"])
    if card["revision"] != cfg["revision"] or card["local_directory"] != cfg["model_directory"]:
        raise ValueError("原始模型来源不匹配")
    for item in card["files"]:
        if not verify_weights and item["name"] == "model.safetensors":
            continue
        path = ROOT / cfg["model_directory"] / item["name"]
        if not path.is_file():
            raise FileNotFoundError(
                f"模型资源缺失：{path}。请先执行 bash scripts/run_phase4_v1.sh prepare {model}，"
                "下载并校验完成后再运行训练。")
        if sha256(path) != item["sha256"]:
            raise ValueError("原始模型文件校验失败")
    sources = ["configs/baselines_extra.yaml", "configs/phase4_v1.yaml", "src/gridpath/training.py", "src/gridpath/training_helpers.py",
               "src/gridpath/evaluation.py", "src/gridpath/inference.py", "src/gridpath/prompts.py",
               "src/gridpath/metrics.py", "src/gridpath/executor.py", "src/gridpath/resources.py",
               "src/gridpath/training_config.py", "scripts/phase4.py", "scripts/run_phase4_v1.sh"]
    payload = {"config":cfg, "dataset":sha256(ROOT / "manifests/dataset_manifest.json"),
               "audit":sha256(ROOT / "results/data_audit.json"), "model":card,
               "sources":{p:original_source_sha(ROOT / p) for p in sources},
               "packages":{p:importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")}}
    lineage_path=ROOT/"manifests/v1_training_lineage.json"
    if lineage_path.exists() and model in read_json(lineage_path)["models"]:
        lineage=read_json(lineage_path)
        if lineage["data_manifest_sha256"]!=payload["dataset"]:
            raise ValueError("修复开发数据不属于当前登记版本")
        entry=lineage["models"][model]
        previous=entry["training_binding"]
        # 仅在训练输入、G100选模输入、优化配置和权重全部未变时沿用训练身份。
        # 当前R100及源码身份独立锁在评估binding，不改写checkpoint历史元数据。
        validate_training_lineage(lineage,model,payload,read_json(ROOT/"manifests/dataset_manifest.json"),dev)
        run=ROOT/cfg["run_directory"]
        completed=read_json(run/"completed.json")
        if (completed["fingerprint"]!=previous["fingerprint"] or completed["state"]["consumed_examples"]!=12000
                or read_json(run/"run.json")!=previous
                or read_json(run/"best/checkpoint.json")["files"]["model.safetensors"]!=entry["model_sha256"]):
            raise ValueError("修复开发修订仅复用原已完成三遍的模型")
        return cfg,train,dev,previous
    return cfg, train, dev, {"fingerprint":digest(payload), **payload}


def validate_training_lineage(lineage,model,payload,manifest,dev):
    previous=lineage["models"][model]["training_binding"]
    if (lineage["repair_dev_revision"]!="invalid100" or payload["config"]!=previous["config"]
            or payload["model"]!=previous["model"] or payload["packages"]!=previous["packages"]
            or payload["sources"]!=lineage["approved_runtime_sources"]):
        raise ValueError("仅允许已登记的修复开发修改，不能更改训练或运行源码")
    for name,expected in lineage["unchanged_files"].items():
        if manifest["files"][name]["sha256"]!=expected:
            raise ValueError("训练或测试数据变化，不能复用原权重")
    generated=[r for r in dev if r["task"]=="GENERATE"]
    if digest(generated)!=lineage["dev_generate_sha256"]:
        raise ValueError("G100选模输入变化，不能沿用已选检查点")
