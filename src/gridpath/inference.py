"""T5严格解码与完整序列评分；不清洗特殊token，不自动截断。"""
from contextlib import nullcontext
from dataclasses import dataclass,replace
import torch
from transformers import StoppingCriteria
from .data_schema import grid_from_record
from .executor import execute,normalize_actions

@dataclass(frozen=True)
class Prediction:
    text: str
    generated_tokens: int
    failure: str | None = None

def autocast_context(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if (
        torch.device(device).type == "cuda" and precision == "bf16") else nullcontext()

def decode_prediction(tokenizer, sequence, cap: int) -> Prediction:
    # T5 的第一个 decoder token 是起始 pad；只去掉协议中的 EOS/pad。
    # 不能使用 skip_special_tokens=True 静默删除模型产生的 <extra_id_*>。
    ids = [int(value) for value in sequence][1:]
    eos = tokenizer.eos_token_id
    ended = eos in ids
    content = ids[:ids.index(eos)] if ended else ids
    if ended and any(token != tokenizer.pad_token_id for token in ids[ids.index(eos) + 1:]):
        return Prediction(tokenizer.decode(content, skip_special_tokens=False), len(content) + 1,
                          "tokens_after_eos")
    text = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    count = len(content) + int(ended)
    failure = None if ended and count <= cap else "token_limit"
    if any(token in tokenizer.all_special_ids for token in content):
        failure = failure or "special_token"
    return Prediction(text, count, failure)

def score_path(row: dict, prediction: Prediction, max_actions: int):
    raw = execute(grid_from_record(row), prediction.text)
    failure = prediction.failure
    if len(normalize_actions(prediction.text)) > max_actions:
        failure = failure or "action_limit"
    # 长度失败是生成接口状态，不冒充执行器四种首错类型。
    effective = replace(raw, success=False, feasible=False) if failure else raw
    return effective, {"interface_failure": failure, "execution": raw.to_dict()}

class BudgetStoppingCriteria(StoppingCriteria):
    def __init__(self, budget):
        self.budget = budget

    def __call__(self, input_ids, scores, **kwargs):
        return self.budget.remaining_seconds < 30
