export CUDA_VISIBLE_DEVICES=6
export JAVA_HOME="/home/wangqiyao/IR-WebQ/configs/jdk/graalvm-jdk-24.0.2+11.1"
export PATH="$JAVA_HOME/bin:$PATH"
export JVM_PATH="/home/wangqiyao/IR-WebQ/configs/jdk/graalvm-jdk-24.0.2+11.1/lib/server/libjvm.so"
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH

EMBEDDINGS=/data/shared/users/wangqiyao/ir-webq/outputs/faiss/bge-m3
ENCODER=/data/shared/users/wangqiyao/models/BAAI/bge-m3

python3 -m pyserini.encode \
  input   --corpus  datasets/candidate_pool.jsonl \
          --fields text \
          --delimiter '\n' \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings $EMBEDDINGS\
          --to-faiss \
  encoder --encoder $ENCODER\
          --fields text \
          --batch 64 \
          --max-length 512 \
          --dimension 1024 \
          --pooling 'cls' \
          --device "cuda:0" \
          --l2-norm \
          --fp16