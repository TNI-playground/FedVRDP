#!/bin/bash

# Array of configurations (modify as per your requirements)
CONFIGS=(
  "attack/fmnist/exp_nondefend/agrTailoredTrmean_image_fmnist_10.yaml"
  "attack/fmnist/exp_nondefend/agrTailoredTrmean_image_fmnist_15.yaml"
  "attack/fmnist/exp_nondefend/agrTailoredTrmean_image_fmnist_20.yaml"
  "attack/fmnist/exp_nondefend/agrTailoredTrmean_image_fmnist_25.yaml"
  "attack/fmnist/exp_nondefend/fang_trmean_median_gray_image_fmnist_10.yaml"
  "attack/fmnist/exp_nondefend/fang_trmean_median_gray_image_fmnist_15.yaml"
  "attack/fmnist/exp_nondefend/fang_trmean_median_gray_image_fmnist_20.yaml"
  "attack/fmnist/exp_nondefend/fang_trmean_median_gray_image_fmnist_25.yaml"
        )

# Function to check GPU usage
function check_gpu_usage() {
  local gpu_util=$(nvidia-smi --id=5 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')
  echo $gpu_util
}

# Function to run code with a configuration
function run_code() {
  local config=$1
  echo "Running code with configuration: $config"
  # Modify the command below to run your code with the specified configuration
  python main.py --gpu 5 --config_name $config
}

# Loop through the configurations
for config in "${CONFIGS[@]}"; do
  # Check GPU usage
  gpu_usage=$(check_gpu_usage)
  
  # Check if GPU is not in use
  run_code "$config"
  sleep 20  # Adjust the sleep duration as needed
done