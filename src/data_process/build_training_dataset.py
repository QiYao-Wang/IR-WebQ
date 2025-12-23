import argparse
import pandas as pd
from tqdm import tqdm
from utils import load_json, save_jsonl


def main(args):
    if args.output_path is None:
        args.output_path = f"../datasets/training_dataset_{args.strategy}.jsonl"
    data = []
    print(f"> Build {args.output_path} from : {args.input_path}")
    df = load_json(args.input_path)
    for sample in df:
        query = {"query": "", "pos": [], "neg": []}
        query["query"] = sample['question']
        for pos in sample["positive_ctxs"]:
            query["pos"].append(pos["text"])
        if args.strategy == 'both':
            for neg in sample['hard_negative_ctxs']:
                query["neg"].append(neg['text'])
        data.append(query)
    # save corpus
    save_jsonl(args.output_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="../datasets/data/webq-train.json"
    )
    parser.add_argument(
        "--strategy", type=str, choices=["both", "only_pos"], default="both"
    )
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
