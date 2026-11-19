#!/usr/bin/env bash
# Run from Cloud Shell or a Bash workstation with gcloud, from any directory.
set -euo pipefail

usage='Usage: PROJECT_ID=... ZONE=... VM=... bash run_cloud_suite.sh smoke|benchmark|long-benchmark|science [seed]'
case ${1:-} in
  smoke)     steps=2;    block=1;  modes=(eager); weights=(0.001) ;;
  benchmark) steps=201;  block=20; modes=(eager fused compile); weights=(0.001 0.001 0.001) ;;
  long-benchmark) steps=1001; block=20; modes=(eager fused compile); weights=(0.001 0.001 0.001) ;;
  science)   steps=6000; block=50; modes=(eager eager); weights=(0.001 0) ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
seed=${2:-1337}
[[ $# -le 2 && "$seed" =~ ^[0-9]{1,9}$ ]] || { echo "$usage" >&2; exit 2; }
: "${PROJECT_ID:?Set PROJECT_ID}" "${ZONE:?Set ZONE}" "${VM:?Set VM}"
flags=(--project="$PROJECT_ID" --zone="$ZONE")
case ${TPU_API:-compute} in
  compute) api=(gcloud compute) ;;
  legacy)  api=(gcloud compute tpus tpu-vm); flags+=(--worker=0) ;;
  *) echo 'TPU_API must be compute or legacy' >&2; exit 2 ;;
esac
remote() { "${api[@]}" ssh "$VM" "${flags[@]}" --command="$1"; }
copy() { "${api[@]}" scp "${flags[@]}" "$@"; }

source_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p cloud-runs
local_dir=$(mktemp -d "cloud-runs/$1-$seed-XXXXXXXX")
group=$(basename "$local_dir")
work="lejepa-sprint/$group"
python='"$HOME/.venvs/lejepa-torchtpu/bin/python"'
echo "Local results: $local_dir. Remote working directory: ~/$work"
remote "test -x $python && mkdir -p lejepa-sprint && mkdir '$work'"
copy "$source_dir/run_experiment.py" "$source_dir/spiral.py" "$VM:$work/"

outcome=0
for i in "${!modes[@]}"; do
  mode=${modes[$i]}
  weight=${weights[$i]}
  name="$mode-w$weight"
  # Only the selected preset and numeric seed enter the remote shell command.
  command="cd '$work' && env -u TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT TORCH_TPU_TIER2_COMPILATION_CACHE=disabled $python run_experiment.py"
  command+=" --device tpu --mode $mode --weight $weight --seed $seed --steps $steps --block-size $block"
  command+=" --batch-size 128 --width 128 --slices 64 --threads 1 --comparison-group '$group' --output '$name'"
  echo "Running $mode, weight $weight, seed $seed for $steps steps"
  if ! remote "$command" 2>&1 | tee "$local_dir/$name.log"; then
    outcome=1
  fi
  mkdir "$local_dir/$name"
  # Preserve each completed record at the controller before starting the next process.
  copy "$VM:$work/$name/result.json" "$local_dir/$name/result.json" || exit 1
done
echo "Retrieved results: $local_dir"
exit "$outcome"
