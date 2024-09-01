#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:a100:2         # Request 2 GPU "generic resources�.
#SBATCH --tasks-per-node=2  # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in
#SBATCH --mem=60G      
#SBATCH --time=00-23:00
#SBATCH --output=%N-%j.out

module load python 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index scikit-learn
pip install python-hostlist
pip install opencv-python
pip install munkres
pip install PyYAML
pip install numpy --no-index

pip install matplotlib scipy
pip install pandas


pip install torch
pip install torchvision
pip install torchaudio

pip install torch_geometric

pip install pytorch_lightning
pip install -r requirements.txt

pip install networkx


export NCCL_BLOCKING_WAIT=1  
export MASTER_ADDR=$(hostname) 

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"


srun python train.py --config config/Avenue/GiCiSAD_train.yaml