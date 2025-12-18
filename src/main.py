import argparse
import os
import pandas as pd
from tqdm import tqdm

from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher
from utils import load_jsonl
from FlagEmbedding import FlagAutoReranker

class Retriever:
    def __init__(self, model_path, index_path, l2_norm=True) -> None:
        self.model_path = model_path
        self.index_path = index_path
        self.l2_norm = l2_norm
        self.encoder = AutoQueryEncoder(model_path,l2_norm=l2_norm)
        self.searcher = FaissSearcher(index_path, self.encoder)

    def retrieve(self, query: str, top_k: int = 200):
        hits = self.searcher.search(query, k=top_k, threads=5)
        return hits

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
    params = {
        "model_path": args.encoder_path,
        "idenx_path": args.index_path,
        "l2_norm": True,
    }
    retriever = Retriever(**params)
    results = []
    for sample in tqdm(samples):
        query = sample["query"]
        hits = retriever.retrieve(query)
        doc_ids = [hit.docid for hit in hits]
        if args.reranker_path is not None:
            reranker = FlagAutoReranker.from_finetuned(
                args.reranker_path,
                query_max_length=256,
                passage_max_length=256,
                device=["auto"],
                model_class="encoder-only-base"
            )
            scores = reranker.compute_score(
                [[query, candidates_df[int(doc_id)]] for doc_id in doc_ids]
            )
            doc_ids = [
                x for x,_ in sorted(zip(doc_ids,scores), key=lambda x:x[1], reverse=True)
            ]
        result = [cal_hit(doc_id, sample["answers"],candidates_df) for doc_id in doc_ids]
        results.append(sum(result)/len(result))