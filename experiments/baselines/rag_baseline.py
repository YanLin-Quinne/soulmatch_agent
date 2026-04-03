"""RAG-only baseline using dialogue chunk retrieval before prediction."""

import math
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.agents.llm_router import router, AgentRole

from .utils import (
    format_dialogue,
    parse_json_response,
    PERSONALITY_PROMPT_TEMPLATE,
    RELATIONSHIP_PROMPT_TEMPLATE,
)

__all__ = ["RAGBaseline"]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "this",
    "to",
    "us",
    "we",
    "were",
    "with",
    "you",
    "your",
    "yours",
}


def _tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


class RAGBaseline:
    """RAG-only baseline: chunk, retrieve, and predict with one LLM call."""

    def __init__(self, top_k: int = 5, chunk_size: int = 3, llm_router=None):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.router = llm_router or router
        self._idf: Dict[str, float] = {}
        self._vocab: List[str] = []
        self._vectorizer = None

    def _chunks_from_dialogue(self, dialogue: List[Dict]) -> List[str]:
        chunks = []
        for idx in range(0, len(dialogue), self.chunk_size):
            chunk = dialogue[idx : idx + self.chunk_size]
            chunks.append(format_dialogue(chunk))
        return chunks

    def _fit_sparse_tfidf(self, texts: List[str]) -> List[List[float]]:
        doc_tokens = [_tokenize(text) for text in texts]
        vocab = sorted({token for tokens in doc_tokens for token in tokens})
        if not vocab:
            self._idf = {}
            self._vocab = []
            return [[0.0] for _ in texts]

        n_docs = len(texts)
        df = {token: 0 for token in vocab}
        for tokens in doc_tokens:
            for token in set(tokens):
                df[token] += 1

        self._idf = {
            token: math.log((1 + n_docs) / (1 + df[token])) + 1.0 for token in vocab
        }
        self._vocab = vocab

        vectors = []
        for tokens in doc_tokens:
            if not tokens:
                vectors.append([0.0] * len(vocab))
                continue

            counts = Counter(tokens)
            total = float(len(tokens))
            vectors.append(
                [
                    (counts.get(token, 0) / total) * self._idf[token]
                    for token in vocab
                ]
            )
        return vectors

    def _embed_query(self, query: str) -> List[float]:
        if self._vectorizer is not None:
            return self._vectorizer.transform([query]).toarray()[0].tolist()

        if not self._vocab:
            return [0.0]

        tokens = _tokenize(query)
        if not tokens:
            return [0.0] * len(self._vocab)

        counts = Counter(tokens)
        total = float(len(tokens))
        return [
            (counts.get(token, 0) / total) * self._idf.get(token, 0.0)
            for token in self._vocab
        ]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        numerator = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return numerator / (norm_a * norm_b)

    def _embed_chunks(self, dialogue: List[Dict]) -> List[Tuple[str, List[float]]]:
        """Split dialogue into chunks and compute TF-IDF vectors."""
        texts = self._chunks_from_dialogue(dialogue)
        if not texts:
            return []

        self._vectorizer = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            self._vectorizer = vectorizer
            vectors = matrix.toarray().tolist()
            return list(zip(texts, vectors))
        except Exception as e:
            logger.debug(f"[RAGBaseline] sklearn TF-IDF unavailable, using fallback: {e}")

        vectors = self._fit_sparse_tfidf(texts)
        return list(zip(texts, vectors))

    def _retrieve(self, query: str, chunks: List[Tuple[str, List[float]]]) -> List[str]:
        """Retrieve the top-k chunks most relevant to the query."""
        if not chunks:
            return []

        query_vec = self._embed_query(query)
        scored = []
        for idx, (text, vector) in enumerate(chunks):
            scored.append((self._cosine_similarity(query_vec, vector), idx, text))

        scored.sort(reverse=True)
        selected = [text for _, _, text in scored[: min(self.top_k, len(scored))]]

        if any(score > 0 for score, _, _ in scored):
            return selected
        return [text for text, _ in chunks[-min(self.top_k, len(chunks)) :]]

    def predict_personality(self, dialogue: List[Dict]) -> Optional[Dict]:
        chunks = self._embed_chunks(dialogue)
        retrieved = self._retrieve(
            "personality traits preferences values habits communication style emotional tendencies",
            chunks,
        )
        dialogue_text = "\n\n".join(
            f"[Retrieved Chunk {i + 1}]\n{text}" for i, text in enumerate(retrieved)
        ) or format_dialogue(dialogue)
        prompt = PERSONALITY_PROMPT_TEMPLATE.format(dialogue=dialogue_text)

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.FEATURE,
                system="You are a personality analysis expert. Use only the retrieved conversation chunks. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[RAGBaseline] personality prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[RAGBaseline] failed to parse personality response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "rag_baseline"
        return result

    def predict_relationship(
        self, dialogue: List[Dict], context: Optional[Dict] = None
    ) -> Optional[Dict]:
        chunks = self._embed_chunks(dialogue)
        retrieved = self._retrieve(
            "relationship dynamics intimacy affection commitment conflict social closeness status",
            chunks,
        )
        dialogue_text = "\n\n".join(
            f"[Retrieved Chunk {i + 1}]\n{text}" for i, text in enumerate(retrieved)
        ) or format_dialogue(dialogue)

        context_section = ""
        if context:
            context_section = f"Additional context:\n{context}"

        prompt = RELATIONSHIP_PROMPT_TEMPLATE.format(
            dialogue=dialogue_text,
            context_section=context_section,
        )

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.GENERAL,
                system="You are a relationship analysis expert. Use only the retrieved conversation chunks. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[RAGBaseline] relationship prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[RAGBaseline] failed to parse relationship response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "rag_baseline"
        return result
