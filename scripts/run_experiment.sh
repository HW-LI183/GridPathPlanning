#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
method="${1:-}"; shift || true
case "$method" in v2) entry=phase4_v2;; improvement_verify_v1) entry=improvement_verify_v1;; *) echo '未知方法'; exit 2;; esac
mode="${1:-}"; shift || true
case "$mode" in prepare|pilot|train|evaluate|test|summarize|select|review|compare) ;; *) echo '用法：模式 [模型 ...|all]；summarize/review/compare [dev|test_id|test_ood] [模型 ...|all]'; exit 2;; esac
if [[ "$method" == v2 && "$mode" == compare ]]; then echo '比较属于改进方法；v2不依赖改进产物'; exit 2; fi
split=dev
if [[ "$mode" == summarize || "$mode" == review || "$mode" == compare ]]; then
 case "${1:-}" in dev|test_id|test_ood) split="$1"; shift;; esac
fi
models=(t5_base t5_small flan_t5_base flan_t5_small)
if [[ $# -gt 0 && "$1" != all ]]; then
 models=()
 for model in "$@"; do
  case "$model" in t5_base|t5_small|flan_t5_base|flan_t5_small) ;; *) echo "未知模型：$model"; exit 2;; esac
  if [[ " ${models[*]} " == *" $model "* ]]; then echo "重复模型：$model"; exit 2; fi
  models+=("$model")
 done
elif [[ $# -gt 1 ]]; then echo 'all不能与其他模型混用'; exit 2
fi
mkdir -p "logs/$method"
run() {
 local exp_log="$1"; shift
 "${GRIDPATH_CONDA:-conda}" run --no-capture-output --name gridpath \
  python "scripts/$entry.py" "$@" > "logs/$method/$exp_log.log" 2>&1
}
case "$mode" in
 prepare|select) run "$mode" "$mode" --models "${models[@]}"; exit ;;
 summarize|review|compare) run "${mode}_${split}" "$mode" --split "$split" --models "${models[@]}"; exit ;;
esac
for model in "${models[@]}"; do
 echo "$method：$mode $model"
 if [[ "$mode" == train || "$mode" == pilot ]]; then
  roles=(verify)
  if [[ "$method" == v2 ]]; then roles=(generate verify); fi
  for role in "${roles[@]}"; do
   if [[ "$mode" == train ]]; then run "${model}_${role}_pilot" pilot --model "$model" --role "$role"; fi
   run "${model}_${role}_${mode}" "$mode" --model "$model" --role "$role"
  done
 elif [[ "$mode" == evaluate ]]; then
  run "${model}_dev" evaluate --model "$model"
 else
  for split in test_id test_ood; do run "${model}_${split}" test --model "$model" --split "$split"; done
 fi
done
if [[ "$mode" == evaluate || "$mode" == test ]]; then
 splits=(dev);if [[ "$mode" == test ]]; then splits=(test_id test_ood);fi
 for split in "${splits[@]}"; do
  run "summarize_${split}" summarize --split "$split" --models "${models[@]}"
  if [[ "$split" == dev ]]; then run select select --models "${models[@]}"; fi
  run "review_${split}" review --split "$split" --models "${models[@]}"
 done
fi
