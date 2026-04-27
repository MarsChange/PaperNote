# PaperNote Agentic RAG Evaluation

This folder contains reproducible retrieval evaluations for the current PaperNote Agentic RAG stack.

## What It Measures

`evaluate_agentic_rag.py` evaluates the retrieval stage used by the chat agent. In batch mode it still evaluates each paper independently by passing the case's own `paper_id` into `vector_store.agentic_retrieve`; it does not merge multiple papers into one global retrieval pool.

- Hybrid dense + sparse retrieval from Milvus when embeddings are available.
- Pure dense-vector baseline through `--retrieval-mode dense_only`.
- Dense fallback and lexical fallback paths through `vector_store.agentic_retrieve`.
- Optional rerank and auto-merge behavior already configured in the backend.
- RAGAS ID-based `context_precision` and `context_recall`.
- Manual retrieval metrics: `hit_rate@k`, `precision@k`, `recall@k`, `mrr`, and average retrieved context count.

The script builds a weakly supervised dataset from existing `agentic_leaf_chunks.json` files. Each sample uses one indexed paper block as the reference context and generates a query from its title, section, caption, semantic metadata, or text. This is useful for regression testing the retriever, but it is not a human-labeled benchmark.

## Run

From the repo root:

```bash
PYTHONPYCACHEPREFIX=/tmp/python-cache PYTHONPATH=backend conda run -n paper-note python experiments/evaluate_agentic_rag.py --limit 24 --top-k 6
```

To evaluate a specific uploaded paper:

```bash
PYTHONPYCACHEPREFIX=/tmp/python-cache PYTHONPATH=backend conda run -n paper-note python experiments/evaluate_agentic_rag.py --paper-id <paper-id> --limit 24 --top-k 6
```

To batch-evaluate all uploaded papers independently:

```bash
PYTHONPYCACHEPREFIX=/tmp/python-cache PYTHONPATH=backend conda run -n paper-note python experiments/evaluate_agentic_rag.py --all-papers --samples-per-paper 24 --top-k 6
```

To run the strict dense-vector baseline:

```bash
PYTHONPYCACHEPREFIX=/tmp/python-cache PYTHONPATH=backend conda run -n paper-note python experiments/evaluate_agentic_rag.py --all-papers --samples-per-paper 24 --top-k 6 --retrieval-mode dense_only
```

If the local backend is already running, Milvus Lite may hold an exclusive lock on the default database file. In that case, copy the database and point the evaluator at the copy:

```bash
cp backend/app/data/milvus/papernote.db /tmp/papernote_eval.db
PYTHONPYCACHEPREFIX=/tmp/python-cache PYTHONPATH=backend conda run -n paper-note python experiments/evaluate_agentic_rag.py --all-papers --samples-per-paper 24 --top-k 6 --milvus-uri /tmp/papernote_eval.db
```

When the backend is already running, prefer using it directly so Milvus retrieval happens inside the existing backend process:

```bash
PYTHONPYCACHEPREFIX=/tmp/python-cache PYTHONPATH=backend conda run -n paper-note python experiments/evaluate_agentic_rag.py --all-papers --samples-per-paper 24 --top-k 6 --backend-url http://127.0.0.1:8000
```

Outputs are written under:

```text
experiments/artifacts/agentic_rag_eval/<timestamp>_<paper_id>/
```

## Dependencies

The current `paper-note` conda environment has been prepared with:

```bash
conda run -n paper-note python -m pip install "ragas>=0.3.0" "datasets>=2.18.0"
```

If the backend provider keys or Milvus index are unavailable, the script still exercises the existing fallback retrieval path because it calls the same `agentic_retrieve` method used by chat.
