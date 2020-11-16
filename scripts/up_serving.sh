#! /bin/bash

RATIO_GPU=${RATIO_GPU:-0.45}
echo "Using GPU: $NVIDIA_VISIBLE_DEVICES"
echo "Limiting GPU to ratio: $RATIO_GPU"

python3 -m mot.serving.app &
/usr/bin/tf_serving_entrypoint.sh --per_process_gpu_memory_fraction=$RATIO_GPU
