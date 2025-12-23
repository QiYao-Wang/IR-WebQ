import argparse
import pandas as pd
from tqdm import tqdm
from src.utils import save_jsonl


def main(args):
    print(f"> Build {args.output_path} from : {args.input_path}")
    data = []
    # load corpus
    df = pd.read_csv(args.input_path, sep="\t")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        data.append({"id": index, "text": row.to_dict()["text"]})
    # save corpus
    save_jsonl(args.output_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="datasets/corpus/wiki_webq_corpus.tsv"
    )
    parser.add_argument(
        "--output_path", type=str, default="datasets/candidate_pool.jsonl"
    )
    args = parser.parse_args()
    main(args)
