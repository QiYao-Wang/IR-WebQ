import argparse
import os
from typing import List, Optional, Set

import pandas as pd
from tqdm import tqdm

from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher
from utils import load_jsonl
from FlagEmbedding import FlagAutoReranker


def parse_methods(methods_raw: str) -> Set[str]:
    return {method.strip() for method in methods_raw.split(",") if method.strip()}


class Retriever:
    def __init__(self, model_path, index_path, l2_norm=True) -> None:
        self.model_path = model_path
        self.index_path = index_path
        self.l2_norm = l2_norm
        self.encoder = AutoQueryEncoder(model_path, l2_norm=l2_norm)
        self.searcher = FaissSearcher(index_path, self.encoder)

    def retrieve(self, query: str, top_k: int = 200):
        hits = self.searcher.search(query, k=top_k, threads=5)
        return hits


def batch_rerank_doc_ids(
    queries: List[str],
    doc_ids_list: List[List[str]],
    candidates_df: List[dict],
    reranker: FlagAutoReranker,
    batch_size: int,
) -> (List[List[str]], List[List[float]]):
    sentence_pairs = []
    offsets = []
    for query, doc_ids in zip(queries, doc_ids_list):
        offsets.append(len(sentence_pairs))
        for doc_id in doc_ids:
            sentence_pairs.append(
                (str(query), str(candidates_df[int(doc_id)]))
            )
    if not sentence_pairs:
        return doc_ids_list, [[] for _ in doc_ids_list]

    scores = reranker.compute_score(sentence_pairs, batch_size=batch_size)

    new_doc_ids_list = []
    new_scores_list = []
    for i, doc_ids in enumerate(doc_ids_list):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < len(offsets) else len(scores)
        slice_scores = scores[start:end]
        sorted_pairs = sorted(zip(doc_ids, slice_scores), key=lambda x: x[1], reverse=True)
        new_doc_ids_list.append([x for x, _ in sorted_pairs])
        new_scores_list.append([s for _, s in sorted_pairs])
    return new_doc_ids_list, new_scores_list


def cal_hit(doc_id, answers, df):
    for answer in answers:
        if answer in df[int(doc_id)]["text"]:
            return 1
    return 0


def main(args):
    test_df = pd.read_csv(args.test_path, sep='\t', names=["query", "answers"])
    samples = []
    for index in range(test_df.shape[0]):
        sample = {
            "query": test_df.iloc[index]["query"],
            "answers": eval(test_df.iloc[index]["answers"])
        }
        samples.append(sample)

    candidates_df = load_jsonl(args.candidates_path)
    methods = parse_methods(args.methods)
    use_mixed = "mixed" in methods
    use_hyde = "hyde" in methods
    use_listwise = "listwise_rerank" in methods
    batch_size = args.batch_size

    if any([use_mixed, use_hyde, use_listwise]):
        from function import (
            build_hyde_query,
            listwise_rerank,
            load_bm25_searcher,
            load_generator,
            mixed_search,
        )

    params = {
        "model_path": args.encoder_path,
        "index_path": args.index_path,
        "l2_norm": True,
    }
    retriever = Retriever(**params)
    reranker: Optional[FlagAutoReranker] = None
    if args.reranker_path is not None:
        reranker = FlagAutoReranker.from_finetuned(
            args.reranker_path,
            query_max_length=256,
            passage_max_length=256,
            device=["auto"],
            model_class="encoder-only-base"
        )

    bm25_searcher = load_bm25_searcher(args.bm25_index_path) if use_mixed else None
    text_generator = None
    generator_model = None
    if use_hyde or use_listwise:
        generator_model = args.hyde_model_path or args.listwise_model_path
        text_generator = load_generator(generator_model)

    results = [] if not args.skip_hit_calc else None
    top_docs = []

    # 预生成检索用 query（HyDE）
    search_queries = [
        build_hyde_query(sample["query"], text_generator) if use_hyde else sample["query"]
        for sample in samples
    ]

    for start in tqdm(range(0, len(samples), batch_size)):
        end = min(len(samples), start + batch_size)
        batch_queries = search_queries[start:end]
        batch_qids = [str(i) for i in range(start, end)]
        batch_samples = samples[start:end]
        raw_queries = [s["query"] for s in batch_samples]

        dense_dict = retriever.searcher.batch_search(
            batch_queries, batch_qids, k=200, threads=5
        )

        batch_doc_ids_list = []
        batch_scores_list = []
        for local_idx, qid in enumerate(batch_qids):
            search_query = batch_queries[local_idx]
            dense_hits = dense_dict.get(qid, [])

            if use_mixed:
                doc_ids, scores = mixed_search(search_query, dense_hits, bm25_searcher)
            else:
                doc_ids = [hit.docid for hit in dense_hits]
                scores = [hit.score for hit in dense_hits]

            batch_doc_ids_list.append(doc_ids)
            batch_scores_list.append(scores)

        if reranker is not None:
            batch_doc_ids_list, batch_scores_list = batch_rerank_doc_ids(
                raw_queries, batch_doc_ids_list, candidates_df, reranker, args.reranker_batch_size
            )

        for local_idx, (doc_ids, scores) in enumerate(zip(batch_doc_ids_list, batch_scores_list)):
            sample = batch_samples[local_idx]

            if use_listwise:
                doc_ids, scores = listwise_rerank(sample["query"], doc_ids, scores, candidates_df, text_generator)

            if results is not None:
                result = [cal_hit(doc_id, sample["answers"], candidates_df) for doc_id in doc_ids]
                results.append(result)

            top_docs.append((doc_ids, scores))

    # 保存检索结果与可选 hit 计算
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_root = os.path.abspath(os.path.join(project_root, "..", "outputs"))
    methods_tag = "_".join(sorted(methods)) if methods else "none"
    prefix_dir = os.path.join(outputs_root, methods_tag)
    os.makedirs(prefix_dir, exist_ok=True)

    # 保存为 pkl 供 cal_hit_multi 使用
    import pickle
    pkl_path = args.save_pkl_path or os.path.join(prefix_dir, "retrieval_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(top_docs, f)

    print(f"> Encoder:{args.encoder_path}")
    print(f"> Reranker:{args.reranker_path}")
    print(f"> Generator:{generator_model if (use_hyde or use_listwise) else 'none'} | loaded:{text_generator is not None}")
    print(f"> Methods:{','.join(sorted(methods)) if methods else 'none'}")
    print(f"> Save to:{prefix_dir}")
    print(f"> PKL saved at:{pkl_path}")

    if not args.skip_hit_calc:
        for k in [1, 10, 100]:
            data = []
            for result in results:
                data.append(1 if sum(result[:k]) > 0 else 0)
            df = pd.DataFrame(
                {
                    "id": range(1, len(data) + 1),
                    "hit": data,
                }
            )
            print(f"{k}\t{sum(data) / len(data)}")
            df.to_csv(os.path.join(prefix_dir, f"hit@{k}.txt"), index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--candidates_path", type=str)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--encoder_path", type=str)
    parser.add_argument("--reranker_path", type=str, default=None)
    parser.add_argument("--methods", type=str, default="", help="mixed,hyde,listwise_rerank (comma separated)") # --methods mixed,hyde
    parser.add_argument("--bm25_index_path", type=str, default=None)
    parser.add_argument("--hyde_model_path", type=str, default=None)
    parser.add_argument("--listwise_model_path", type=str, default=None)
    parser.add_argument("--save_pkl_path", type=str, default=None, help="path to save retrieval results pkl")
    parser.add_argument("--skip_hit_calc", action="store_true", help="skip hit@k calculation for speed")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for retrieval")
    parser.add_argument("--reranker_batch_size", type=int, default=1024, help="batch size for reranker scoring")
    args = parser.parse_args()
    main(args)

