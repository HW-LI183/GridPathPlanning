#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_experiment.sh" improvement_verify_v1 "$@"
