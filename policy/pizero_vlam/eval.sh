#!/bin/bash
# Usage: ./eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <ckpt_path> [execute_steps]
#
#   ckpt_setting   free-form label; only used to name the eval_result/ output directory
#   ckpt_path      absolute path to the barrel checkpoint file (see deploy_policy.yml)
#
# Example:
#   ./eval.sh beat_block_hammer demo_clean qwen3vl-15k 0 0 \
#       /path/to/session/checkpoints/15000.pt

policy_name=pizero_vlam
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
ckpt_path=${6}
execute_steps=${7:-5}

if [ -z "${ckpt_path}" ]; then
    echo -e "\033[31mckpt_path (argument 6) is required\033[0m"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --ckpt_path ${ckpt_path} \
    --execute_steps ${execute_steps}
