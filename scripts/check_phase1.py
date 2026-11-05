#!/usr/bin/env python3
"""检查阶段 1 的 Conda、CUDA、tokenizer 和本地模型文件，不训练。"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
from prepare_models import save as save_model_manifest, verify_file


def main():
    check_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-models", action="store_true", help="要求两个模型权重均已下载并能加载")
    args = parser.parse_args()
    import torch
    from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Config)
    import datasets
    import matplotlib
    import sklearn
    import yaml

    project = json.loads((ROOT / "configs/project.json").read_text())
    model_manifest = json.loads((ROOT / "manifests/models.json").read_text())
    if [entry["repo_id"] for entry in model_manifest["models"]] != project["models"]:
        raise RuntimeError("固定模型清单与项目决策不一致")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "python_executable": sys.executable, "python_version": platform.python_version(),
              "conda_prefix": sys.prefix, "platform": platform.platform(), "packages": {},
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
              "upstream_commit": subprocess.check_output(
                  ["git", "-C", str(ROOT / "third_party/ppnl_upstream"), "rev-parse", "HEAD"],
                  text=True).strip(), "models": []}
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
        text=True)
    report["nvidia_smi"] = hardware.strip().splitlines()
    if report["upstream_commit"] != project["upstream_commit"]:
        raise RuntimeError("上游 HEAD 与项目固定版本不一致")
    requirements = ["torch", "transformers", "tokenizers", "huggingface-hub", "accelerate",
                    "sentencepiece", "protobuf", "datasets", "numpy", "scikit-learn",
                    "matplotlib", "PyYAML", "pytest"]
    for name in requirements:
        report["packages"][name] = importlib.metadata.version(name)
    if not report["cuda_available"]:
        raise RuntimeError("当前进程无法访问 CUDA；请在工作站终端执行检查")
    report["gpu_name"] = torch.cuda.get_device_name(0)
    report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    report["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
    report["bf16_supported"] = torch.cuda.is_bf16_supported()
    started = time.monotonic()
    with torch.inference_mode():
        tensor = torch.ones((16, 16), device="cuda")
        assert torch.all(tensor @ tensor == 16).item()
        if report["bf16_supported"]:
            value = tensor.to(torch.bfloat16)
            assert torch.isfinite(value @ value).all().item()
        torch.cuda.synchronize()
    report["gpu_probe_seconds"] = round(time.monotonic() - started, 6)
    del tensor
    report["gpu_probe_description"] = "仅16x16 FP32/BF16算子检查，无训练或模型批量评估"

    # 仅构造 Trainer，核实 Transformers/Accelerate API 能衔接，不调用 train()。
    with tempfile.TemporaryDirectory(prefix="gridpath-phase1-") as temporary:
        tiny_config = T5Config(vocab_size=128, d_model=32, d_ff=64, num_layers=1,
                               num_heads=2, decoder_start_token_id=0)
        tiny_model = AutoModelForSeq2SeqLM.from_config(tiny_config)
        arguments = Seq2SeqTrainingArguments(output_dir=temporary, use_cpu=True, report_to=[])
        trainer = Seq2SeqTrainer(model=tiny_model, args=arguments)
        report["training_api_constructed_without_training"] = True
        del trainer, tiny_model

    complete = True
    for metadata in model_manifest["models"]:
        repo_id = metadata["repo_id"]
        directory = ROOT / "models" / repo_id.rsplit("/", 1)[-1]
        if metadata["local_directory"] != str(directory.relative_to(ROOT)):
            raise RuntimeError("模型清单中的本地目录不匹配")
        verified = []
        for artifact in metadata["files"]:
            local_file = directory / artifact["name"]
            if local_file.exists():
                artifact["sha256"] = verify_file(local_file, artifact)
                artifact["downloaded"] = True
                verified.append(artifact["name"])
            else:
                artifact["downloaded"] = False
                if artifact["name"] != "model.safetensors":
                    raise RuntimeError(f"缺少清单文件：{local_file}")
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        example = "GENERATE The grid has 6 rows and 6 columns. Start: (0, 0). Goal: (5, 5)."
        sample = tokenizer(example, truncation=False)
        entry = {"repo_id": repo_id, "directory": str(directory.relative_to(ROOT)),
                 "revision": metadata["revision"], "verified_files": verified,
                 "model_type": config.model_type, "d_model": config.d_model,
                 "tokenizer_class": type(tokenizer).__name__,
                 "tokenizer_model_max_length": tokenizer.model_max_length,
                 "example_tokens": len(sample["input_ids"]), "weights_loaded": False}
        if (directory / "model.safetensors").exists():
            started = time.monotonic()
            model = AutoModelForSeq2SeqLM.from_pretrained(directory, local_files_only=True,
                                                         use_safetensors=True)
            entry["parameter_count"] = sum(p.numel() for p in model.parameters())
            entry["weights_loaded"] = True
            entry["cpu_load_seconds"] = round(time.monotonic() - started, 3)
            del model
        else:
            complete = False
        report["models"].append(entry)
    report["phase1_acceptance"] = "complete" if complete else "pending_model_weights"
    save_model_manifest(model_manifest)
    manifest_dir = ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements-lock.txt").write_text(freeze)
    report["check_wall_seconds"] = round(time.monotonic() - check_started, 6)
    report["resource_note"] = "按完整检查命令的wall time保守登记GPU占用；gpu_probe_seconds仅为算子段。"
    (manifest_dir / "runtime.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_models and not complete:
        raise SystemExit("配置及 tokenizer 已通过；请先执行 prepare_models.py 下载两个权重，再重做最终验收。")


if __name__ == "__main__":
    main()
