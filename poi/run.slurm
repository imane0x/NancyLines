#!/bin/bash
#SBATCH --job-name="POI"
#SBATCH --output=/lustre/fswork/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --constraint=a100
#SBATCH --account=knb@a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=2:00:00

module purge
module load arch/a100
#module load miniforge/25.9.1
#conda activate osm
 module load uv/0.8.3
source /lustre/fsn1/projects/rech/knb/ukq43aj/osm/.venv/bin/activate

set -x
export WANDB_MODE=offline

srun python make_poi_datasets.py 
srun python jsonl_to_txt.py --show_answer
#srun python make_gps_traces_datasets.py 
#srun python train.py
#srun python eval.py --model_type llm --model_path /lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-4B-Instruct-2507
