<!--
 * @Descripttion: hanoiy‘s code
 * @version: 1.0.0
 * @Author: hanoiy
 * @Date: 2025-12-23 17:35:38
 * @LastEditors: hanoiy
 * @LastEditTime: 2025-12-28 15:11:07
-->
# IR-WebQ: Homework of IR Class in UCAS

## Quick Start

```bash
git clone https://github.com/QiYao-Wang/IR-WebQ.git

cd IR-WebQ

export PYTHONPATH=$PWD

mkdir datasets # copy datasets here
mkdir configs/jdk # copy jdk here

conda create -n ir-webq python=3.10

pip install -r requirements.txt 

wget https://download.oracle.com/graalvm/24/latest/graalvm-jdk-24_linux-x64_bin.tar.gz
tar -zxvf graalvm-jdk-24_linux-x64_bin.tar.gz
mv -r "graalvm file" $PWD/configs/jdk

python -m spacy download en_core_web_sm

conda install -y -c conda-forge \
  faiss-gpu=1.7.4 \
  cudatoolkit=11.8 \
  nomkl
```

1. Candidates Construction
```bash
sh scripts/candidate_data.sh --input_path $Your-Corpus-Path --output_path $Your-Candidates-Path
```

2. Indexing the corpus

Note: this step need jdk. You need place the jdk in "config/jdk/jdk-your-version"
```bash
sh scripts/index.sh
```

3. Run the Main Program
The `--methods` parameter supports three options: `mixed`, `hyde`, `listwise_rerank` (comma separated).
```bash
sh scripts/main.sh --reranker_model_path --methods "mixed,hyde"
```
If you need reranker, please provide the reranker model path in `main.sh`.

## SFT Reranker Fine-tuning (Optional)

The following steps are optional and only needed if you want to fine-tune the reranker model.

4. Build Training Dataset
```bash
sh scripts/build_training_dataset.sh
```

5. Hard Negative Mining
```bash
sh scripts/build_candidate_pool.sh
sh scripts/hn_mine.sh
```

6. Fine-tune Reranker
```bash
sh scripts/bge_reranker_finetuning.sh --training_data_path $Your-Training-Dataset-Path
```

7. Run Main Program with Reranker and Evaluate
```bash
sh scripts/main.sh --reranker_model_path $Your-Reranker-Model-Path --methods "mixed,hyde"
``` 
