export CUDA_VISIBLE_DEVICES="5"
export PYTHONPATH="/data/shared/yangzhihao/miniconda3/envs/ir-webq/lib/python3.10/site-packages:$PYTHONPATH"
export DISABLE_FLASH_ATTN=1
export WANDB_DISABLED=true
export MASTER_ADDR=localhost
export MASTER_PORT=12349
export WORLD_SIZE=1

STRATEGY=both
NEGATIVE_NUMBER=15
TRAIN_DATA=datasets/training_dataset_${STRATEGY}_minedHN_${NEGATIVE_NUMBER}.jsonl
OUTPUT_DIR=outputs/reranker_ft_${STRATEGY}_minedHN_${NEGATIVE_NUMBER}_v1
LOG_DIR=logs/bge-reranker-v2-m3_${STRATEGY}_minedHN_${NEGATIVE_NUMBER}.log
mkdir -p logs

/home/yangzhihao/.local/bin/torchrun \
    --nproc_per_node=1 \
    --master_port=12349 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    --model_name_or_path /data/shared/users/wangqiyao/models/BAAI/bge-reranker-v2-m3 \
    --train_data $TRAIN_DATA \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 50 \
    --dataloader_num_workers 2 \
    --report_to none \
    2>&1 | tee $LOG_DIR
