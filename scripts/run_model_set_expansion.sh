#!/usr/bin/env bash
# 已测试模型集合的追加流程；不训练，真实开发/测试推理由用户启动。
set -euo pipefail
cd "$(dirname "$0")/.."
method="${1:-}"; shift || true
case "$method" in v2) entry=phase4_v2;; improvement_verify_v1) entry=improvement_verify_v1;; *) echo '用法：bash scripts/run_model_set_expansion.sh v2|improvement_verify_v1 [模型 ...|all]'; exit 2;; esac
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
  python "$@" > "logs/$method/$exp_log.log" 2>&1
}
# 逐个评价，不调用常规evaluate包装器末尾的旧集合select。
for model in "${models[@]}"; do
 echo "$method：核验或补齐dev $model"
 run "${model}_dev" "scripts/$entry.py" evaluate --model "$model"
done
run summarize_dev "scripts/$entry.py" summarize --split dev --models "${models[@]}"
run expand_model_set scripts/expand_model_set.py --method "$method" --models "${models[@]}" --apply
run select "scripts/$entry.py" select --models "${models[@]}"
run review_dev "scripts/$entry.py" review --split dev --models "${models[@]}"
echo "$method：开发集合核验完成，继续单错/双错测试"
bash scripts/run_experiment.sh "$method" test "${models[@]}"
