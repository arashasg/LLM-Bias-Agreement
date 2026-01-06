#!/bin/bash
#SBATCH --job-name=generation_model_job
#SBATCH --gres=gpu:2   # Dynamically use NUM_OF_GPUS or default to 2 GPUs
#SBATCH --time=16:00:00                 # Set the time limit (adjust as needed)
#SBATCH -c 8                            # Request 4 CPU cores
#SBATCH --mem=50G                       # Request 50 GB of memory
#SBATCH --output=slurm-%j.out           # Log the output to a file with job ID
#SBATCH --error=slurm-%j.err            # Log the errors to a file with job ID
#SBATCH --exclude=lse-hpcnode9   # Exclude nodes lse-hpcnode6 and lse-hpcnode9
#SBATCH --qos=gpu2-16h
#SBATCH --signal=B:USR1@120            # Receive SIGUSR1 signal 120 seconds before job ends

# Ensure the model name is passed as the first argument
if [ -z "$1" ]; then
    echo "Error: Model name is required as the first argument."
    exit 1
fi

MODEL_NAME=$1




# Activate the conda environment
source activate stereoset
# export CUDA_LAUNCH_BLOCKING=1



# Run the Python script in the background
python run_generation_model.py --model "$MODEL_NAME" &

# Capture the PID of the background process
PID=$!

# Trap signal to restart the job 120 seconds before the job ends
trap 'bash run.sh $MODEL_NAME' SIGUSR1

# Wait for the background process to finish
wait $PID
