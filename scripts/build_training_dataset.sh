PYTHON_BIN="/data/shared/yangzhihao/miniconda3/envs/ir-webq/bin/python"
export PYTHONPATH="/home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/src:${PYTHONPATH}"

$PYTHON_BIN /home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/src/data_process/build_training_dataset.py \
    --input_path /home/yangzhihao/shared/users/wangqiyao/ir-webq/IR_2025_Project/datas/webq-train.json \
    --strategy both \
    --output_path /home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ/datasets/training_dataset_both.jsonl