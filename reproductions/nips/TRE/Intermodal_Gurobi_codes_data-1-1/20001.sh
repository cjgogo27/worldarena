#!/bin/bash
#SBATCH --account=innovation
#SBATCH --job-name=gurobi-1-1
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1
#SBATCH --ntasks=3                         # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G                           # Job memory request
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=UAV-GV/test1.log
module load gurobi/11.0.1
python3 main.py & rm -r /scratch/yimengzhang/cjyang/Intermodal_Gurobi_codes_data-1-1/Figures/experiment606250001 
wait
