#!/bin/bash

# Ensure the arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <MODEL_NAME> <NUM_OF_GPUS>"
  exit 1
fi

MODEL_NAME=$1
NUM_OF_GPUS=$2

# Call the .slurm file with the parameters passed as arguments
sbatch generation_model_job_updated_bug.slurm $MODEL_NAME $NUM_OF_GPUS