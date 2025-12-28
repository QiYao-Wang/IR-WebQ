export CUDA_VISIBLE_DEVICES=5
export JAVA_HOME="/home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/configs/jdk/graalvm-jdk-24.0.2+11.1"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Use the correct python from ir-webq environment
/data/shared/yangzhihao/miniconda3/envs/ir-webq/bin/python src/main.py \
    --test_path /data/shared/users/wangqiyao/ir-webq/IR_2025_Project/datas/webq-test.csv \
    --candidates_path datasets/candidate_pool.jsonl \
    --index_path /data/shared/users/wangqiyao/ir-webq/outputs/faiss/bge-m3 \
    --encoder_path /data/shared/users/wangqiyao/models/BAAI/bge-m3 \
    --reranker_path /data/shared/users/wangqiyao/models/BAAI/bge-reranker-v2-m3 \
    --methods "hyde" \
    --bm25_index_path /home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/outputs/lucene/bm25-webq \
    --hyde_model_path /home/yangzhihao/shared/models/meta-llama/Llama-3.1-8B-Instruct \
    --listwise_model_path /home/yangzhihao/shared/models/meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 128 \

# --save_pkl_path /home/yangzhihao/data1_link/UCAS_hmw_term1/IR/outputs/norerank/retrieval_results.pkl \
# --reranker_path /data/shared/users/wangqiyao/models/BAAI/bge-reranker-v2-m3
# --reranker_path /home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/outputs/reranker_ft_both_minedHN_100/checkpoint-2474 \