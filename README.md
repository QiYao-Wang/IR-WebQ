# IR-WebQ

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
```

### Data Processing
1. Candidates Construction
```bash
sh scripts/candidate_data.sh --input_path $Your-Corpus-Path --output_path $Your-Candidates-Path
```

2. Indexing the corpus

Note: this step need jdk. You need place the jdk in "config/jdk/jdk-your-version"
```bash
sh scripts/index.sh
```