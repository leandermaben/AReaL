#!/bin/sh
#SBATCH --mem=480g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=gpuH200x8
#SBATCH --time=3:00:00
#SBATCH --job-name=server_qwen80b
#SBATCH --account=bbjs-delta-gpu 
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=verbose,closest
#SBATCH --error=/u/lmaben/speech/long_speech/LongAudioUnderstandingSystem/agent_search/vllm_server/launch/job_outputs/%x__%j.err
#SBATCH --output=/u/lmaben/speech/long_speech/LongAudioUnderstandingSystem/agent_search/vllm_server/launch/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lmaben@andrew.cmu.edu    

cd /u/lmaben/speech/long_speech/LongAudioUnderstandingSystem/agent_search/vllm_server
source vllm_server_env/bin/activate
source /u/lmaben/agent_safety/AReaL/examples/agent_safety/.env

export HF_HOME=/tmp/lmaben/hf_home_dir
export HF_DATASETS_CACHE=/tmp/lmaben/hf_cache_dir

export TMPDIR=/tmp/lmaben/tmp
export TRITON_CACHE_DIR=/tmp/lmaben/triton_cache_$SLURM_JOB_ID
mkdir -p $TRITON_CACHE_DIR


vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --port 8000 --tensor-parallel-size 2 --max-model-len 24000 --gpu-memory-utilization 0.9

