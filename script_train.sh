#!/bin/bash
#SBATCH --job-name=train          # Job name
#SBATCH --ntasks=1                    # Run on a single GPU
#SBATCH --time=00:30:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END

cd /projects/0/nisei0750/nadia/repo/hep_transformer/

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

python -m pip install --upgrade pip
python -m pip install torch==2.1.2
#python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
python -m pip install scikit-learn
python -m pip install pandas

# python -m pip install packaging
# python -m pip install ninja
# python -m pip install flash-attn --no-build-isolation

python training_f16.py