#!/bin/bash
#SBATCH --job-name=generation_model_job
#SBATCH --gres=gpu:2   # Dynamically use NUM_OF_GPUS or default to 2 GPUs
#SBATCH --time=3:59:00                 # Set the time limit (adjust as needed)
#SBATCH -c 16                            # Request 4 CPU cores
#SBATCH --mem=50G                       # Request 50 GB of memory
#SBATCH --output=slurm-%j.out           # Log the output to a file with job ID
#SBATCH --error=slurm-%j.err            # Log the errors to a file with job ID
#SBATCH --exclude=lse-hpcnode9   # Exclude nodes lse-hpcnode6 and lse-hpcnode9
#SBATCH --qos=gpu8-4h
#SBATCH --signal=B:USR1@120            # Receive SIGUSR1 signal 120 seconds before job ends

# Ensure the model name is passed as the first argument

if [ -z "$1" ]; then
    echo "Error: Model name is required as the first argument."
    exit 1
fi

MODEL_NAME=$1
NUM_OF_GPUS=$2

# Validate NUM_OF_GPUS (it should be a valid integer)
if ! [[ "$NUM_OF_GPUS" =~ ^[0-9]+$ ]]; then
    echo "Error: NUM_OF_GPUS must be an integer."
    exit 1
fi

# Activate the conda environment
source activate stereoset
# export CUDA_LAUNCH_BLOCKING=1



# Run the Python script in the background
python run_generation_model_reddit.py  "$MODEL_NAME" &

# Capture the PID of the background process
PID=$!

# Trap signal to restart the job 120 seconds before the job ends
trap 'bash run_reddit.sh $MODEL_NAME $NUM_OF_GPUS' SIGUSR1

# Wait for the background process to finish
wait $PID