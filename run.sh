# batch_run/models/SCOPE_BAN/run.sh
#!/bin/bash

# Get the absolute path of the current script
MODEL_DIR=$(dirname $(realpath $0))

CONFIG_PATH=$MODEL_DIR/config_yaml/default.yaml
# CONFIG_PATH=$MODEL_DIR/config_yaml/test.yaml
if [ -n "$6" ]; then
    CONFIG_PATH=$6
fi
OUTPUT_DIR=$1
SPLIT_PATH=$2
TENSORBOARD_LOGDIR=$3 # ignored here, tensorboard included in the output dir
SEED=$4
TRAIN_PATH=$SPLIT_PATH/train_3d.parquet
VAL_PATH=$SPLIT_PATH/val_3d.parquet
TEST_PATH=$SPLIT_PATH/test_3d.parquet
CUDA_VISIBLE_DEVICES=$5

# using yacs

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 $MODEL_DIR/main.py $CONFIG_PATH \
    DATA.TRAIN $TRAIN_PATH DATA.VAL $VAL_PATH DATA.TEST $TEST_PATH \
    RESULT.OUTPUT_DIR $OUTPUT_DIR SOLVER.SEED $SEED 

# example of running the model
# bash run.sh result_test_batch /share/home/grp-huangxd/chenyigang/DTI-Project/data/runs/20240921_114807/split -1 0 3 config_yaml/test.yaml