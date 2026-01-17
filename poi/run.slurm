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
#SBATCH --time=0:10:00

module purge
module load arch/a100
module load miniforge/25.9.1
conda activate osm
# module load uv/0.8.3
# source /lustre/fswork/projects/rech/knb/ukq43aj/git/NancyLines/poi/.venv/bin/activate

set -x
export WANDB_MODE=offline

#srun python make_pois_datasets.py 
srun python make_gps_traces_datasets.py 
#srun python train.py
