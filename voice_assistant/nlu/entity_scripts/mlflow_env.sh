#!/bin/bash
export MLFLOW_EXPERIMENT_NAME="workout_ner"
export MLFLOW_RUN_NAME="roberta_tok2vec_transparser_v1"
export MLFLOW_TRACKING_URI="file:./mlruns"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

