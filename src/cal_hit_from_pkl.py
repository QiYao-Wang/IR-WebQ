import argparse
import os
import pickle
from typing import Dict, Iterable, List

import pandas as pd

from utils import load_jsonl


def load_retrieval_results(pkl_path: str) -> List[tuple]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_answers(test_path: str) -> List[List[str]]:
    df = pd.read_csv(test_path, sep="\t", names=["query", "answers"])
    return [eval(item) if isinstance(item, str) else [] for item in df["answers"].tolist()]


def cal_hit(doc_id: str, answers: Iterable[str], candidates: List[dict]) -> int:
    text = candidates[int(doc_id)]["text"]
    for answer in answers:
        if answer in text:
            return 1
    return 0


def compute_hits(
    top_docs: List[tuple],
    answers_list: List[List[str]],
    candidates: List[dict],
    k_list: List[int],
) -> Dict[int, List[int]]:
    if len(top_docs) != len(answers_list):
        raise ValueError(f"样本数量不一致: top_docs={len(top_docs)}, answers={len(answers_list)}")

    hits = {k: [] for k in k_list}
    for idx, (doc_ids, _) in enumerate(top_docs):
        answers = answers_list[idx]
        for k in k_list:
            hit_value = 1 if any(cal_hit(doc_id, answers, candidates) for doc_id in doc_ids[:k]) else 0
            hits[k].append(hit_value)
    return hits


def save_hits(hits: Dict[int, List[int]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for k, values in hits.items():
        df = pd.DataFrame({"id": range(1, len(values) + 1), "hit": values})
        out_path = os.path.join(output_dir, f"sft_v1_hit@{k}.txt")
        df.to_csv(out_path, index=False, sep="\t")
        acc = sum(values) / len(values) if values else 0
        print(f"hit@{k}: {acc:.4f} -> {out_path}")


def parse_k_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    pkl_path = os.path.abspath(args.pkl_path)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(pkl_path)
    candidates_path = os.path.abspath(args.candidates_path) if args.candidates_path else os.path.join(
        project_root, "datasets", "candidate_pool.jsonl"
    )
    test_path = os.path.abspath(args.test_path)
    k_list = parse_k_list(args.k) if isinstance(args.k, str) else [1, 10, 100]

    top_docs = load_retrieval_results(pkl_path)
    candidates = load_jsonl(candidates_path)
    answers_list = load_answers(test_path)

    hits = compute_hits(top_docs, answers_list, candidates, k_list)
    save_hits(hits, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从检索结果 pkl 计算 hit@k (默认 1,10,100)")
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="/home/yangzhihao/data1_link/UCAS_hmw_term1/IR/outputs/none/retrieval_results.pkl",
        help="检索结果 pkl 路径 (top_docs 列表)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/data/shared/users/wangqiyao/ir-webq/IR_2025_Project/datas/webq-test.csv",
        help="包含 query 与答案的测试集 (制表符分隔)",
    )
    parser.add_argument(
        "--candidates_path",
        type=str,
        default=None,
        help="候选文档 jsonl 路径，默认使用项目 datasets/candidate_pool.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出 hit@k txt 的目录，默认与 pkl 同级",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,10,100",
        help="逗号分隔的 k 列表，例如 1,5,10",
    )
    args = parser.parse_args()
    main(args)

