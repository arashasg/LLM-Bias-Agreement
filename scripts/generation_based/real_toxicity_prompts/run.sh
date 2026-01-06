#!/bin/bash


MODEL_NAME=$1


# Call the .slurm file with the parameters passed as arguments
# sbatch generation_model_job_updated_llama.slurm $MODEL_NAME $NUM_OF_GPUS
sbatch generation_model_job_updated.slurm $MODEL_NAME 