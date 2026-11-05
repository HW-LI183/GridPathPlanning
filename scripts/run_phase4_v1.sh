#!/usr/bin/env bash
# 用户执行：v1两个原始T5，4000G/三遍，G100/R100开发评估。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
v1_mode=${1:-all}
case "$v1_mode" in all|train|evaluate|test) ;; *) echo "用法：all、train、evaluate、test" >&2; exit 2 ;; esac
v1_conda=${GRIDPATH_CONDA:-/home/user/miniconda3/bin/conda}
mkdir -p logs/v1
"$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py preflight >> logs/v1/preflight.log 2>&1
if [[ "$v1_mode" == all || "$v1_mode" == train ]]; then
  for v1_model in t5_base t5_small; do
    echo "v1：$v1_model pilot与最多三遍训练"
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py pilot --model "$v1_model" >> "logs/v1/${v1_model}_pilot.log" 2>&1
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py train --model "$v1_model" >> "logs/v1/${v1_model}_train.log" 2>&1
  done
  "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py select-primary >> logs/v1/select_primary.log 2>&1
fi
if [[ "$v1_mode" == all || "$v1_mode" == evaluate ]]; then
  for v1_model in t5_base t5_small; do
    echo "v1开发集三条流程：$v1_model"
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py evaluate --model "$v1_model" >> "logs/v1/${v1_model}_dev.log" 2>&1
  done
  "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py summarize >> logs/v1/summary_dev.log 2>&1
fi
if [[ "$v1_mode" == test ]]; then
  # 不在all中自动启动最终测试。需已有开发集选模锁，测试不用于调参。
  for v1_split in test_id test_ood; do
    for v1_model in t5_base t5_small; do
      "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py evaluate --model "$v1_model" --split "$v1_split" >> "logs/v1/${v1_model}_${v1_split}.log" 2>&1
    done
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py summarize --split "$v1_split" >> "logs/v1/summary_${v1_split}.log" 2>&1
  done
fi
echo "完成：results/v1/summary_<划分>.json"
