"""阶段四v1的通用训练与原子快照工具。"""
import os
import random
import shutil
import uuid
from pathlib import Path
import numpy as np
import torch
from .inference import autocast_context
from .resources import utc_now
from .training_config import read_json, write_json, sha256

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def stream_indices(count: int, seed: int, consumed: int, batch_size: int) -> list[int]:
    epoch, offset = divmod(consumed, count)
    indices = list(range(count))
    random.Random(seed + epoch).shuffle(indices)
    return indices[offset:min(offset + batch_size, count)]

def collate(features: list[dict], pad_id: int, device) -> dict:
    width = max(len(row["input_ids"]) for row in features)
    target_width = max(len(row["labels"]) for row in features)
    inputs = torch.full((len(features), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    labels = torch.full((len(features), target_width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(features):
        size = len(row["input_ids"])
        inputs[index, :size] = torch.tensor(row["input_ids"], device=device)
        mask[index, :size] = 1
        labels[index, :len(row["labels"])] = torch.tensor(row["labels"], device=device)
    return {"input_ids": inputs, "attention_mask": mask, "labels": labels}

def train_update(model, optimizer, scheduler, features, tokenizer, config):
    """一个有效 batch；按非 padding 目标 token 数加权微批损失。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    target_tokens = sum(len(row["labels"]) for row in features)
    loss_sum = 0.0
    for start in range(0, len(features), config["micro_batch_size"]):
        batch = collate(features[start:start + config["micro_batch_size"]], tokenizer.pad_token_id, device)
        count = int((batch["labels"] != -100).sum())
        with autocast_context(device, config["precision"]):
            loss = model(**batch, use_cache=False).loss
            weighted = loss * (count / target_tokens)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练 loss 非有限，立即停止且不重放样本")
        weighted.backward()
        loss_sum += float(weighted.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {"loss": loss_sum, "grad_norm": float(norm), "learning_rate": scheduler.get_last_lr()[0]}

def rng_state(*, include_cuda: bool = False) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if include_cuda else None}

def restore_rng(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])

def save_checkpoint(run: Path, aliases: list[str], model, tokenizer, optimizer, scheduler,
                    state: dict, binding: dict):
    storage = run / "_states"
    storage.mkdir(parents=True, exist_ok=True)
    destination = storage / uuid.uuid4().hex
    destination.mkdir()
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_state(include_cuda=next(model.parameters()).is_cuda), "state": state},
               destination / "training_state.pt")
    metadata = {"schema_version": "gridpath.checkpoint.v1", "created_at": utc_now(),
                "state": state, "binding": binding,
                "files": {path.name: sha256(path) for path in destination.iterdir() if path.is_file()}}
    write_json(destination / "checkpoint.json", metadata)
    for alias in aliases:
        link = run / f".{alias}.next"
        link.unlink(missing_ok=True)
        link.symlink_to(destination.relative_to(run), target_is_directory=True)
        os.replace(link, run / alias)
    # 三个软链接可共享同一快照；只删除已无引用的本程序存储目录。
    referenced = {(run / alias).resolve() for alias in ("latest", "pilot", "best") if (run / alias).exists()}
    for path in storage.iterdir():
        if path.is_dir() and len(path.name) == 32 and path.resolve() not in referenced:
            shutil.rmtree(path)
    return destination

def validate_checkpoint(path: Path, expected_fingerprint: str | None = None) -> dict:
    metadata = read_json(path / "checkpoint.json")
    fingerprint = metadata["binding"]["fingerprint"]
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("续训配置、数据或运行源码与 checkpoint 不同")
    for name, expected in metadata["files"].items():
        if Path(name).name != name or sha256(path / name) != expected:
            raise ValueError(f"checkpoint 文件校验失败：{name}")
    return metadata

def load_training_state(path, optimizer, scheduler, device):
    # 只用于本地生成且哈希校验通过的 checkpoint，包含 NumPy/Python RNG。
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and key != "step":
                values[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return payload["state"]

def require_gpu(config):
    if not torch.cuda.is_available():
        raise RuntimeError("正式入口需要 CUDA；CPU 软件验证请运行 python -m pytest")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 不可用，停止；不要自动切换为 FP16")
