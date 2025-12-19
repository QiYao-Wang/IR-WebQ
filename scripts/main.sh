export CUDA_VISIBLE_DEVICES=6

python src/main.py \
    --test_path /data/shared/users/wangqiyao/ir-webq/IR_2025_Project/datas/webq-test.csv \
    --candidates_path datasets/candidate_pool.jsonl \
    --index_path /data/shared/users/wangqiyao/ir-webq/outputs/faiss/bge-m3 \
    --encoder_path /data/shared/users/wangqiyao/models/BAAI/bge-m3 \
    --reranker_path /data/shared/users/wangqiyao/models/BAAI/bge-reranker-v2-m3
