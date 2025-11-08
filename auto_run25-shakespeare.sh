#!/bin/bash

# Array of configurations (modify as per your requirements)
CONFIGS=(
  # "attack/shakespeare/exp_attacknum25/agrTailoredMedian_text_shakespeare_median.yaml"
  # "attack/shakespeare/exp_attacknum25/agrTailoredTrmean_text_shakespeare_dpfed_NM14.yaml"
  # "attack/shakespeare/exp_attacknum25/agrTailoredTrmean_text_shakespeare_epfed_NM14_CB5_CR2.yaml"
  "attack/shakespeare/exp_attacknum25/agrTailoredTrmean_text_shakespeare_flame.yaml"
  # "attack/shakespeare/exp_attacknum25/agrTailoredTrmean_text_shakespeare_sparsefed.yaml"
  # "attack/shakespeare/exp_attacknum25/agrTailoredTrmean_text_shakespeare_tr_mean.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_trmean_median_gray_text_shakespeare_dpfed_NM14.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_trmean_median_gray_text_shakespeare_epfed_NM14_CB5_CR2.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_trmean_median_gray_text_shakespeare_flame.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_trmean_median_gray_text_shakespeare_median.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_trmean_median_gray_text_shakespeare_sparsefed.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_trmean_median_gray_text_shakespeare_tr_mean.yaml"
  # "attack/shakespeare/exp_attacknum25/agrTailoredKrumBulyan_text_shakespeare_krum.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_krum_bulyan_gray_text_shakespeare_krum.yaml"
  # "attack/shakespeare/exp_attacknum25/agrTailoredKrumBulyan_text_shakespeare_bulyan.yaml"
  # "attack/shakespeare/exp_attacknum25/fang_krum_bulyan_gray_text_shakespeare_bulyan.yaml"
        )

# Function to check GPU usage
function check_gpu_usage() {
  local gpu_util=$(nvidia-smi --id=4 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')
  echo $gpu_util
}

# Function to run code with a configuration
function run_code() {
  local config=$1
  echo "Running code with configuration: $config"
  # Modify the command below to run your code with the specified configuration
  python main.py --gpu 4 --config_name $config
}

# Loop through the configurations
for config in "${CONFIGS[@]}"; do
  # Check GPU usage
  gpu_usage=$(check_gpu_usage)
  
  # Check if GPU is not in use
  run_code "$config"
  sleep 20  # Adjust the sleep duration as needed
done
