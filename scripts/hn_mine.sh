export CUDA_VISIBLE_DEVICES=3,4,5
PYTHON_BIN="/data/shared/yangzhihao/miniconda3/envs/ir-webq/bin/python"
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-0}
export PYTHONPATH="/home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/src:${PYTHONPATH}"

NEGATIVE_NUMBER=15
DATASET_DIR="/home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/datasets"

INPUT_FILE="${DATASET_DIR}/training_dataset_both.jsonl"
CANDIDATE_POOL="${DATASET_DIR}/candidate_pool.jsonl"
OUTPUT_FILE="${DATASET_DIR}/training_dataset_both_minedHN_${NEGATIVE_NUMBER}.jsonl"

"${PYTHON_BIN}" /home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/src/data_process/hn_mine.py \
    --embedder_name_or_path /data/shared/users/wangqiyao/models/BAAI/bge-m3 \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --candidate_pool "${CANDIDATE_POOL}" \
    --range_for_sampling 2-200 \
    --negative_number "${NEGATIVE_NUMBER}" \
    --use_gpu_for_searching

# python data_process/hn_mine.py \
#     --embedder_name_or_path /data1/HF-Models/BAAI/bge-large-en-v1.5 \
#     --input_file ../datasets/training_dataset_both.jsonl \
#     --output_file ../datasets/training_dataset_both_minedHN.jsonl \
#     --candidate_pool ../datasets/candidate_pool.jsonl \
#     --range_for_sampling 2-200 \
#     --negative_number 15 \
#     --use_gpu_for_searching