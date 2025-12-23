import re
from typing import Iterable, List, Optional

from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, pipeline


def load_bm25_searcher(index_path: Optional[str]) -> Optional[LuceneSearcher]:
    """Create a BM25 searcher if the index path is provided."""
    if index_path is None:
        return None
    try:
        return LuceneSearcher(index_path)
    except Exception:
        return None


def load_generator(model_path: Optional[str]):
    """Build a text-generation pipeline; supports 本地/远端模型."""
    if model_path is None:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            model_kwargs={"pad_token_id": pad_id},
        )
        if pad_id is not None:
            pipe.model.config.pad_token_id = pad_id
            pipe.tokenizer.pad_token_id = pad_id
        return pipe
    except Exception as e:
        print(f"[WARN] load_generator failed for {model_path}: {e}")
        return None


def mixed_search(
    query: str,
    dense_hits: Iterable,
    bm25_searcher: Optional[LuceneSearcher],
    bm25_k: int = 200,
    rrf_k: int = 60,
) -> (List[str], List[float]):
    """Hybrid retrieval via RRF fusion of dense and BM25 results."""
    dense_hits = list(dense_hits)
    if bm25_searcher is None:
        return [hit.docid for hit in dense_hits], [hit.score for hit in dense_hits]

    bm25_hits = bm25_searcher.search(query, k=bm25_k)
    scores = {}

    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.docid] = scores.get(hit.docid, 0) + 1 / (rrf_k + rank)

    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit.docid] = scores.get(hit.docid, 0) + 1 / (rrf_k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    doc_ids = [doc_id for doc_id, _ in fused]
    fused_scores = [score for _, score in fused]
    return doc_ids, fused_scores


def build_hyde_query(
    query: str,
    generator,
    max_new_tokens: int = 96,
) -> str:
    """Generate a hypothetical answer as the retrieval query."""
    if generator is None:
        return query

    prompt = (
        "Write a concise passage that directly answers the question.\n"
        f"Question: {query}\n"
        "Passage:"
    )
    try:
        output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = output[0].get("generated_text", "").strip()
        return text or query
    except Exception:
        return query


def listwise_rerank(
    query: str,
    doc_ids: List[str],
    scores: List[float],
    candidates_df: List[dict],
    generator,
    top_n: int = 10,
    max_new_tokens: int = 128,
) -> (List[str], List[float]):
    """Listwise rerank the top documents with a small LLM."""
    if generator is None or not doc_ids:
        return doc_ids, scores

    top_candidates = doc_ids[:top_n]
    doc_block = "\n".join(
        [f"[{idx + 1}] {candidates_df[int(doc_id)]['text']}" for idx, doc_id in enumerate(top_candidates)]
    )
    prompt = (
        f"Query: {query}\n"
        "Here are candidate documents:\n"
        f"{doc_block}\n"
        "Return the indices of the most relevant documents in order, separated by spaces or commas."
    )
    try:
        response = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = response[0].get("generated_text", "")
        ranked = _extract_indices(text, len(top_candidates))
        ordered_ids = [top_candidates[i - 1] for i in ranked if 1 <= i <= len(top_candidates)]
        ordered_scores = [scores[doc_ids.index(doc_id)] for doc_id in ordered_ids]
        seen = set(ordered_ids)
        remain = [(doc, scores[doc_ids.index(doc)]) for doc in doc_ids if doc not in seen]
        remain_ids = [doc for doc, _ in remain]
        remain_scores = [score for _, score in remain]
        return ordered_ids + remain_ids, ordered_scores + remain_scores
    except Exception:
        return doc_ids, scores


def _extract_indices(text: str, limit: int) -> List[int]:
    """Parse indices like 1,2,3 or [1] [2] [3] from model output."""
    tokens = re.split(r"[,\s]+", text)
    indices = []
    for token in tokens:
        token = token.strip("[]().")
        if token.isdigit():
            value = int(token)
            if 1 <= value <= limit and value not in indices:
                indices.append(value)
    return indices

