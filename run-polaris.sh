#!/bin/bash -le
#PBS -l select=10:system=polaris
#PBS -l walltime=0:15:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -N test-run
#PBS -A CSC249ADCD08

# Change to working directory
cd ${PBS_O_WORKDIR}

# Remove old run
rm -r test-run runinfo

# Activate the environment
module load conda
conda activate /lus/grand/projects/CSC249ADCD08/multi-rl-workflow/env

# Start redis
redis-server --bind 0.0.0.0 --appendonly no &> redis.log &
redis_pid=$!

# Launch the app
python run_colmena.py

# Clean up
kill $redis_pid
