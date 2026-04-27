"""Persistent BM25 sparse vectors for PaperNote Agentic RAG."""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path

from app.core.config import settings


class BM25SparseEncoder:
    """Incremental BM25 state shared by indexing and retrieval."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = Path(state_path or settings.bm25_state_path)
        self.k1 = 1.5
        self.b = 0.75
        self._lock = threading.RLock()
        self._vocab: dict[str, int] = {}
        self._doc_freq: Counter[str] = Counter()
        self._vocab_counter = 0
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0
        self._load_state()

    @property
    def total_docs(self) -> int:
        return self._total_docs

    def tokenize(self, text: str) -> list[str]:
        value = str(text or "").lower()
        tokens: list[str] = []
        index = 0
        while index < len(value):
            char = value[index]
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
                index += 1
                continue
            match = re.match(r"[a-z0-9][a-z0-9_\-]{1,}", value[index:])
            if match:
                tokens.append(match.group(0))
                index += len(match.group(0))
                continue
            index += 1
        return tokens

    def increment_add_documents(self, texts: list[str]):
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                self._sum_token_len += len(tokens)
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]):
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                self._sum_token_len = max(0, self._sum_token_len - len(tokens))
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()

    def encode(self, text: str) -> dict[int, float]:
        with self._lock:
            sparse, changed = self._encode_unlocked(text)
            if changed:
                self._persist_unlocked()
            return sparse

    def encode_many(self, texts: list[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        with self._lock:
            vectors: list[dict[int, float]] = []
            changed_any = False
            for text in texts:
                sparse, changed = self._encode_unlocked(text)
                vectors.append(sparse)
                changed_any = changed_any or changed
            if changed_any:
                self._persist_unlocked()
            return vectors

    def _encode_unlocked(self, text: str) -> tuple[dict[int, float], bool]:
        tokens = self.tokenize(text)
        if not tokens:
            return {}, False
        tf = Counter(tokens)
        doc_len = len(tokens)
        avg_len = max(self._avg_doc_len, 1.0)
        total_docs = max(self._total_docs, 0)
        sparse: dict[int, float] = {}
        changed = False

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                changed = True
            index = self._vocab[token]
            df = self._doc_freq.get(token, 0)
            idf = (
                math.log((total_docs + 1) / 1)
                if df == 0
                else math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            )
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            score = idf * numerator / max(denominator, 1e-9)
            if score > 0:
                sparse[index] = float(score)
        return sparse, changed

    def _load_state(self):
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if payload.get("version") != 1:
            return
        self._vocab = {str(key): int(value) for key, value in payload.get("vocab", {}).items()}
        self._doc_freq = Counter(
            {str(key): int(value) for key, value in payload.get("doc_freq", {}).items()}
        )
        self._total_docs = int(payload.get("total_docs", 0) or 0)
        self._sum_token_len = int(payload.get("sum_token_len", 0) or 0)
        self._vocab_counter = max(self._vocab.values(), default=-1) + 1
        self._recompute_avg_len()

    def _persist_unlocked(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        temp_path = self.state_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.state_path)

    def _recompute_avg_len(self):
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )


bm25_encoder = BM25SparseEncoder()
