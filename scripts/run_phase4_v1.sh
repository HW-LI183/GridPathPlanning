#!/usr/bin/env bash
# 四基线统一入口；prepare下载、preflight预检、train训练、select选模、evaluate开发评价。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
v1_mode=${1:-all}
v1_target=${2:-all}
case "$v1_mode" in prepare|preflight|all|train|select|evaluate|test) ;; *) echo "用法：$0 {prepare|preflight|train|select|evaluate|test|all} [all|模型键]" >&2; exit 2 ;; esac
case "$v1_target" in
  all) v1_models=(t5_base t5_small flan_t5_base flan_t5_small) ;;
  t5_base|t5_small|flan_t5_base|flan_t5_small) v1_models=("$v1_target") ;;
  flan) v1_models=(flan_t5_base flan_t5_small) ;; # 兼容已有命令
  t5) v1_models=(t5_base t5_small) ;;
  *) echo "模型须为all、t5_base、t5_small、flan_t5_base或flan_t5_small" >&2; exit 2 ;;
esac
v1_conda=${GRIDPATH_CONDA:-/home/user/miniconda3/bin/conda}
mkdir -p logs/v1
if [[ "$v1_mode" == prepare ]]; then
  for v1_model in "${v1_models[@]}"; do
    echo "准备模型资源：$v1_model（工作站直连HF镜像，日志logs/v1/${v1_model}_prepare.log）"
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      "$v1_conda" run --no-capture-output --name gridpath python scripts/fetch_pinned_models.py --model "$v1_model" \
      2>&1 | tee -a "logs/v1/${v1_model}_prepare.log"
  done
  exit 0
fi
if [[ "$v1_mode" == select ]]; then
  "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py select-primary \
    2>&1 | tee -a logs/v1/select_primary_four.log
  exit 0
fi
if ! "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py preflight --models "${v1_models[@]}" >> logs/v1/preflight.log 2>&1; then
  tail -n 18 logs/v1/preflight.log >&2
  echo "预检未通过，尚未启动训练/评价。资源缺失时请先运行：bash scripts/run_phase4_v1.sh prepare $v1_target" >&2
  exit 1
fi
if [[ "$v1_mode" == preflight ]]; then
  echo "CPU预检通过：${v1_models[*]}"
  exit 0
fi
if [[ "$v1_mode" == all || "$v1_mode" == train ]]; then
  for v1_model in "${v1_models[@]}"; do
    echo "v1：$v1_model pilot与最多三遍训练"
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py pilot --model "$v1_model" >> "logs/v1/${v1_model}_pilot.log" 2>&1
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py train --model "$v1_model" >> "logs/v1/${v1_model}_train.log" 2>&1
  done
  # 只有四模型均完成才能形成全体选择；单模型训练不会因其他模型未训练而报错。
  v1_complete=true
  for v1_model in t5_base t5_small flan_t5_base flan_t5_small; do
    [[ -f "checkpoints/v1/$v1_model/completed.json" ]] || v1_complete=false
  done
  if [[ "$v1_complete" == true ]]; then
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py select-primary >> logs/v1/select_primary_four.log 2>&1
  else
    echo "当前模型训练完成；四模型完成后运行 bash scripts/run_phase4_v1.sh select。"
  fi
fi
if [[ "$v1_mode" == all || "$v1_mode" == evaluate ]]; then
  for v1_model in "${v1_models[@]}"; do
    echo "v1开发集三条流程：$v1_model"
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py evaluate --model "$v1_model" >> "logs/v1/${v1_model}_dev.log" 2>&1
  done
  "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py summarize >> logs/v1/summary_dev.log 2>&1
fi
if [[ "$v1_mode" == test ]]; then
  for v1_split in test_id test_ood; do
    for v1_model in "${v1_models[@]}"; do
      "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py evaluate --model "$v1_model" --split "$v1_split" >> "logs/v1/${v1_model}_${v1_split}.log" 2>&1
    done
    "$v1_conda" run --no-capture-output --name gridpath python scripts/phase4.py summarize --split "$v1_split" >> "logs/v1/summary_${v1_split}.log" 2>&1
  done
fi
echo "完成：results/v1/summary_<划分>.json；完整状态以complete字段为准"
