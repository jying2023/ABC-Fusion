#!/bin/bash
MAX_EPOCH=8
LR=2e-5
ADAPTER_LR=1e-4
BATCH_SIZE=32
UPDATE_FREQ=2
CANDIDATES=20
INJECT_POSTION=1
BERT_PATH='pretrained_model/'
TARIN_PATH='train.txt'
WARMUP_RATIO=0.1

# EXP_NAME=confusion_relu_${MAX_EPOCH}_${LR}_${BATCH_SIZE}_${UPDATE_FREQ}_${WARMUP_RATIO}
EXP_NAME=${INJECT_POSTION}confusion${CANDIDATES}_special_lr_${ADAPTER_LR}_const_lr_${MAX_EPOCH}_${LR}_${BATCH_SIZE}_${UPDATE_FREQ}_${WARMUP_RATIO}
OUTPUT_PATH=outputs/$EXP_NAME/
mkdir $OUTPUT_PATH

python main_confusion.py \
    --train_file  $TARIN_PATH \
    --model_name_or_path $BERT_PATH \
    --train_batch_size $BATCH_SIZE \
    --valid_batch_size $BATCH_SIZE \
    --learning_rate $LR \
     --adapter_learning_rate $ADAPTER_LR \
    --num_train_epochs $MAX_EPOCH \
    --candidates_num $CANDIDATES \
    --inject_position $INJECT_POSTION \
    --gradient_accumulation_steps $UPDATE_FREQ \
    --warmup_portion $WARMUP_RATIO \
    --output_dir $OUTPUT_PATH;

