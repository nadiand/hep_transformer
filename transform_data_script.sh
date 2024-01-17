#!/bin/bash
#SBATCH --job-name=data_creation          # Job name
#SBATCH --ntasks=1                    # Run on a single GPU
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END

cd /projects/0/nisei0750/nadia/repo/hep_transformer/

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0

pip install torch==1.13.1
pip install scikit-learn
pip install pandas

# for i in {21001..21005} ; do
#     python domain_decomposition.py -e ${i}
# done

python domain_decomposition.py -e '21001'