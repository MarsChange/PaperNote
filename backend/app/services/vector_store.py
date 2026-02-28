"""Embedding and vector store service using ChromaDB."""

import logging
import re
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
import openai

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document chunking, embedding, and retrieval via ChromaDB."""

    def __init__(self):
        self._client: Optional[chromadb.ClientAPI] = None
        self._openai: Optional[openai.OpenAI] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.Client(
                ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(settings.chroma_dir),
                    anonymized_telemetry=False,
                )
            )
        return self._client

    @property
    def openai_client(self) -> openai.OpenAI:
        if self._openai is None:
            self._openai = openai.OpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )
        return self._openai

    def _get_collection(self, paper_id: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            name=f"paper_{paper_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def chunk_markdown(self, markdown: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
        """
        Split markdown into semantic chunks.
        Tries to split on headings/paragraphs first, falls back to size-based splitting.
        """
        # Split on markdown headings or double newlines
        sections = re.split(r'\n(?=#{1,3}\s)|\n{2,}', markdown)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        current = ""
        for section in sections:
            if len(current) + len(section) <= chunk_size:
                current = f"{current}\n\n{section}".strip() if current else section
            else:
                if current:
                    chunks.append(current)
                # If a single section exceeds chunk_size, split it further
                if len(section) > chunk_size:
                    words = section.split()
                    buf = ""
                    for word in words:
                        if len(buf) + len(word) + 1 > chunk_size:
                            chunks.append(buf)
                            # Keep overlap
                            overlap_words = buf.split()[-overlap // 5:] if overlap else []
                            buf = " ".join(overlap_words + [word])
                        else:
                            buf = f"{buf} {word}".strip()
                    if buf:
                        current = buf
                    else:
                        current = ""
                else:
                    current = section

        if current:
            chunks.append(current)

        return [
            {"text": chunk, "index": i}
            for i, chunk in enumerate(chunks)
        ]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from OpenAI API."""
        if not texts:
            return []

        # Batch in groups of 100
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            resp = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings

    def index_paper(self, paper_id: str, markdown: str):
        """Chunk, embed, and store a paper's content."""
        chunks = self.chunk_markdown(markdown)
        if not chunks:
            logger.warning(f"No chunks generated for paper {paper_id}")
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)

        collection = self._get_collection(paper_id)
        collection.add(
            ids=[f"{paper_id}_chunk_{c['index']}" for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"paper_id": paper_id, "chunk_index": c["index"]} for c in chunks],
        )
        logger.info(f"Indexed {len(chunks)} chunks for paper {paper_id}")

    def search(self, paper_id: str, query: str, top_k: int = 5) -> list[str]:
        """Search for relevant chunks given a query."""
        query_embedding = self.embed_texts([query])
        if not query_embedding:
            return []

        collection = self._get_collection(paper_id)
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        documents = results.get("documents", [[]])
        return documents[0] if documents else []

    def delete_paper(self, paper_id: str):
        """Remove a paper's vector collection."""
        try:
            self.client.delete_collection(f"paper_{paper_id}")
        except Exception:
            pass


vector_store = VectorStore()
