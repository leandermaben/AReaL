#!/bin/sh
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4
#SBATCH --time=8:00:00
#SBATCH --job-name=werewolf_rl_4gpu
#SBATCH --account=bbjs-delta-gpu
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest
#SBATCH --error=/u/lmaben/agent_safety/AReaL/examples/agent_safety/rl/launch/job_outputs/%x__%j.err
#SBATCH --output=/u/lmaben/agent_safety/AReaL/examples/agent_safety/rl/launch/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lmaben@andrew.cmu.edu

cd /u/lmaben/agent_safety/AReaL
source .venv/bin/activate

export HF_HOME=/tmp/lmaben/hf_home_dir
export HF_DATASETS_CACHE=/tmp/lmaben/hf_cache_dir

export TMPDIR=/tmp/lmaben/tmp
export TRITON_CACHE_DIR=/tmp/lmaben/triton_cache_$SLURM_JOB_ID
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR

python -m examples.agent_safety.rl.train \
    --config examples/agent_safety/rl/config_4gpu.yaml
