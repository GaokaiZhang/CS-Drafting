#!/bin/bash

export HF_HOME="/data/user_data/$USER/.hf_cache"
export HF_HUB_CACHE="/data/hf_cache/hub"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"
export HF_LOCAL_MODEL_ROOT="/data/user_data/$USER/hf_models"
export HF_HUB_OFFLINE=1

mkdir -p "$HF_HOME"
mkdir -p "$HF_LOCAL_MODEL_ROOT"
