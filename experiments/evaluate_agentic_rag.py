#!/usr/bin/env python3
"""Evaluate PaperNote Agentic RAG retrieval with RAGAS ID-based metrics."""

from __future__ import annotations

import argparse
import csv
import httpx
import json
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.services.agentic_docstore import agentic_docstore  # noqa: E402
from app.services.vector_store import vector_store  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
FIGURE_RE = re.compile(r"\b(?:fig(?:ure)?\.?)\s*([0-9]+[A-Za-z]?)", re.IGNORECASE)
EQUATION_RE = re.compile(r"(?:\\tag\s*\{([^}]+)\}|equation[-_\s]*([0-9]+))", re.IGNORECASE)
DEFAULT_BLOCK_TYPES = ("text", "image", "table", "equation")
_RAGAS_CACHE: tuple[Any, Any, Any] | None = None
STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "also",
    "among",
    "and",
    "are",
    "battery",
    "between",
    "both",
    "cell",
    "cells",
    "data",
    "dataset",
    "datasets",
    "different",
    "during",
    "each",
    "figure",
    "from",
    "have",
    "into",
    "learning",
    "method",
    "model",
    "more",
    "paper",
    "prediction",
    "results",
    "section",
    "study",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "with",
}


@dataclass
class EvalCase:
    case_id: str
    paper_id: str
    query: str
    reference_context_ids: list[str]
    reference_chunk_id: str
    block_type: str
    page_number: int
    section: str
    title: str
    reference_text_preview: str


@dataclass
class EvalResult:
    case_id: str
    paper_id: str
    query: str
    block_type: str
    page_number: int
    reference_context_ids: str
    retrieved_context_ids: str
    hit: int
    first_hit_rank: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ragas_context_precision: float | None
    ragas_context_recall: float | None
    retrieval_mode: str
    candidate_k: int
    rerank_applied: bool
    auto_merge_applied: bool
    retrieval_error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current PaperNote Agentic RAG retriever with RAGAS.",
    )
    parser.add_argument("--paper-id", default="", help="Uploaded paper id. Defaults to the latest indexed paper.")
    parser.add_argument(
        "--all-papers",
        action="store_true",
        help="Batch-evaluate every uploaded paper independently; retrieval remains scoped to each case's paper_id.",
    )
    parser.add_argument("--limit", type=int, default=24, help="Maximum generated evaluation cases.")
    parser.add_argument(
        "--samples-per-paper",
        type=int,
        default=0,
        help="When --all-papers is set, sample this many cases from each paper. Defaults to --limit per paper.",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Retriever top_k passed to agentic_retrieve.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["agentic", "dense_only"],
        default="agentic",
        help="agentic runs the full current retriever; dense_only runs pure dense vector top-k retrieval.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed.")
    parser.add_argument(
        "--block-types",
        default=",".join(DEFAULT_BLOCK_TYPES),
        help="Comma-separated block types to sample, e.g. text,image,table,equation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "experiments" / "artifacts" / "agentic_rag_eval"),
        help="Directory for JSONL/CSV/summary outputs.",
    )
    parser.add_argument(
        "--milvus-uri",
        default="",
        help="Optional Milvus Lite database path for evaluation, useful when the live backend owns the default DB lock.",
    )
    parser.add_argument(
        "--backend-url",
        default="",
        help="Optional running backend URL. Uses /api/evaluation/retrieve so retrieval runs inside the live backend process.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS scoring and only write manual metrics.",
    )
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def latest_paper_id() -> str:
    paths = indexed_leaf_paths()
    if not paths:
        raise SystemExit(
            f"No indexed papers found under {settings.upload_dir}. Upload and index a paper first."
        )
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0].parents[1].name


def indexed_leaf_paths() -> list[Path]:
    return [
        path
        for path in settings.upload_dir.glob("*/parsed/agentic_leaf_chunks.json")
        if path.is_file()
    ]


def indexed_paper_ids() -> list[str]:
    return sorted(path.parents[1].name for path in indexed_leaf_paths())


def load_leaves(paper_id: str) -> list[dict[str, Any]]:
    leaves = agentic_docstore.load_leaves(paper_id)
    if not leaves:
        raise SystemExit(f"No agentic leaf chunks found for paper_id={paper_id}.")
    return leaves


def extract_keywords(*texts: str, limit: int = 4) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in TOKEN_RE.findall(normalize_text(text)):
            lowered = token.lower().strip("-_")
            if len(lowered) < 4 or lowered in STOPWORDS:
                continue
            counts[lowered] += 1
    return [token for token, _ in counts.most_common(limit)]


def clean_title(title: str, max_length: int = 120) -> str:
    title = normalize_text(title)
    return title[:max_length].rstrip() if len(title) > max_length else title


def figure_label(title: str, block_id: str) -> str:
    match = FIGURE_RE.search(title)
    if match:
        return f"Fig. {match.group(1)}"
    match = re.search(r"image-([0-9]+)$", block_id)
    return f"image block {match.group(1)}" if match else "the figure"


def equation_label(text: str, block_id: str) -> str:
    match = EQUATION_RE.search(text)
    if match:
        return f"公式 {match.group(1) or match.group(2)}"
    match = re.search(r"equation-([0-9]+)$", block_id)
    return f"公式块 {match.group(1)}" if match else "公式"


def make_query(leaf: dict[str, Any]) -> str:
    block_type = str(leaf.get("type") or "text")
    block_id = str(leaf.get("block_id") or leaf.get("id") or "")
    title = clean_title(str(leaf.get("title") or ""))
    section = clean_title(str(leaf.get("section") or ""))
    text = normalize_text(leaf.get("text") or leaf.get("content") or "")
    semantic_summary = normalize_text(leaf.get("semantic_summary") or "")
    metadata = leaf.get("semantic_metadata") if isinstance(leaf.get("semantic_metadata"), dict) else {}
    metadata_summary = normalize_text(metadata.get("summary") if isinstance(metadata, dict) else "")
    keywords = extract_keywords(title, section, semantic_summary, metadata_summary, text)
    topic = "、".join(keywords[:3]) if keywords else (section or title or "相关内容")

    if block_type == "image":
        label = figure_label(title or text, block_id)
        if topic:
            return f"论文中的{label}展示了哪些关于{topic}的信息？"
        return f"论文中的{label}展示了什么内容？"
    if block_type == "table":
        return f"论文在{section or title or '相关部分'}的表格中汇总了哪些关于{topic}的结果？"
    if block_type == "equation":
        return f"论文中{equation_label(text, block_id)}在{section or title or '相关部分'}里表达了什么？"
    if section and topic:
        return f"论文在{section}部分关于{topic}说明了什么？"
    return f"这篇论文中关于{topic}的内容是什么？"


def candidate_quality(leaf: dict[str, Any]) -> int:
    text = normalize_text(leaf.get("text") or leaf.get("content") or "")
    semantic_summary = normalize_text(leaf.get("semantic_summary") or "")
    title = normalize_text(leaf.get("title") or "")
    score = min(len(text), 1000)
    if semantic_summary:
        score += 400
    if title:
        score += 120
    if leaf.get("type") in {"image", "table", "equation"}:
        score += 200
    return score


def build_cases(
    paper_id: str,
    leaves: list[dict[str, Any]],
    block_types: set[str],
    limit: int,
    seed: int,
    case_prefix: str = "",
) -> list[EvalCase]:
    by_block: dict[str, dict[str, Any]] = {}
    for leaf in leaves:
        block_type = str(leaf.get("type") or "text")
        block_id = str(leaf.get("block_id") or leaf.get("id") or "")
        text = normalize_text(leaf.get("text") or leaf.get("content") or "")
        if block_type not in block_types or not block_id or not text:
            continue
        if block_type == "text" and len(text) < 120:
            continue
        current = by_block.get(block_id)
        if current is None or candidate_quality(leaf) > candidate_quality(current):
            by_block[block_id] = leaf

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for leaf in by_block.values():
        grouped[str(leaf.get("type") or "text")].append(leaf)

    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
        group.sort(key=candidate_quality, reverse=True)

    selected: list[dict[str, Any]] = []
    type_order = [item for item in DEFAULT_BLOCK_TYPES if item in block_types]
    while any(grouped.get(block_type) for block_type in type_order):
        for block_type in type_order:
            if grouped.get(block_type):
                selected.append(grouped[block_type].pop(0))
                if limit > 0 and len(selected) >= limit:
                    break
        if limit > 0 and len(selected) >= limit:
            break

    cases: list[EvalCase] = []
    for index, leaf in enumerate(selected, start=1):
        block_id = str(leaf.get("block_id") or leaf.get("id") or "")
        text = normalize_text(leaf.get("text") or leaf.get("content") or "")
        cases.append(
            EvalCase(
                case_id=f"{case_prefix}case-{index:03d}",
                paper_id=paper_id,
                query=make_query(leaf),
                reference_context_ids=[block_id],
                reference_chunk_id=str(leaf.get("chunk_id") or leaf.get("id") or ""),
                block_type=str(leaf.get("type") or "text"),
                page_number=int(leaf.get("page_number") or 0),
                section=normalize_text(leaf.get("section") or ""),
                title=normalize_text(leaf.get("title") or ""),
                reference_text_preview=text[:500],
            )
        )
    if not cases:
        raise SystemExit("No evaluation cases could be generated from the selected block types.")
    return cases


def unique_context_ids(sources: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for source in sources:
        context_id = normalize_text(source.get("block_id") or source.get("id") or "")
        if context_id and context_id not in seen:
            ids.append(context_id)
            seen.add(context_id)
    return ids


def score_ids(retrieved_ids: list[str], reference_ids: list[str]) -> tuple[int, int, float, float, float]:
    reference_set = set(reference_ids)
    retrieved_set = set(retrieved_ids)
    overlap = reference_set & retrieved_set
    first_hit_rank = 0
    for rank, context_id in enumerate(retrieved_ids, start=1):
        if context_id in reference_set:
            first_hit_rank = rank
            break
    hit = int(first_hit_rank > 0)
    precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(overlap) / len(reference_set) if reference_set else 0.0
    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
    return hit, first_hit_rank, precision, recall, mrr


def load_ragas() -> tuple[Any, Any, Any]:
    global _RAGAS_CACHE
    if _RAGAS_CACHE is not None:
        return _RAGAS_CACHE

    try:
        from ragas import SingleTurnSample
    except Exception:
        from ragas.dataset_schema import SingleTurnSample

    try:
        from ragas.metrics.collections import IDBasedContextPrecision, IDBasedContextRecall
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Importing IDBasedContext.*",
                category=DeprecationWarning,
            )
            from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    _RAGAS_CACHE = (SingleTurnSample, IDBasedContextPrecision, IDBasedContextRecall)
    return _RAGAS_CACHE


def ragas_score(
    retrieved_ids: list[str],
    reference_ids: list[str],
    query: str,
) -> tuple[float | None, float | None]:
    SingleTurnSample, IDBasedContextPrecision, IDBasedContextRecall = load_ragas()
    sample = SingleTurnSample(
        user_input=query,
        retrieved_context_ids=retrieved_ids,
        reference_context_ids=reference_ids,
    )
    precision = IDBasedContextPrecision().single_turn_score(sample)
    recall = IDBasedContextRecall().single_turn_score(sample)
    return float(precision), float(recall)


def retrieve_case(
    case: EvalCase,
    top_k: int,
    retrieval_mode: str,
    backend_url: str = "",
) -> dict[str, Any]:
    if not backend_url:
        if retrieval_mode == "dense_only":
            return vector_store.agentic_dense_only_retrieve(
                case.paper_id,
                case.query,
                top_k=top_k,
            )
        return vector_store.agentic_retrieve(case.paper_id, case.query, top_k=top_k)

    endpoint = backend_url.rstrip("/") + "/api/evaluation/retrieve"
    with httpx.Client(timeout=300) as client:
        response = client.post(
            endpoint,
            json={
                "paper_id": case.paper_id,
                "query": case.query,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
            },
        )
        response.raise_for_status()
        return response.json()


def evaluate_cases(
    cases: list[EvalCase],
    top_k: int,
    use_ragas: bool,
    retrieval_mode: str,
    backend_url: str = "",
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        retrieval = retrieve_case(
            case,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            backend_url=backend_url,
        )
        sources = retrieval.get("sources") or []
        meta = retrieval.get("meta") or {}
        retrieved_ids = unique_context_ids(sources)
        hit, first_hit_rank, precision, recall, mrr = score_ids(
            retrieved_ids,
            case.reference_context_ids,
        )
        ragas_precision: float | None = None
        ragas_recall: float | None = None
        if use_ragas:
            ragas_precision, ragas_recall = ragas_score(
                retrieved_ids,
                case.reference_context_ids,
                case.query,
            )
        results.append(
            EvalResult(
                case_id=case.case_id,
                paper_id=case.paper_id,
                query=case.query,
                block_type=case.block_type,
                page_number=case.page_number,
                reference_context_ids=";".join(case.reference_context_ids),
                retrieved_context_ids=";".join(retrieved_ids),
                hit=hit,
                first_hit_rank=first_hit_rank,
                precision_at_k=precision,
                recall_at_k=recall,
                mrr=mrr,
                ragas_context_precision=ragas_precision,
                ragas_context_recall=ragas_recall,
                retrieval_mode=str(meta.get("retrieval_mode") or ""),
                candidate_k=int(meta.get("candidate_k") or 0),
                rerank_applied=bool(meta.get("rerank_applied")),
                auto_merge_applied=bool(meta.get("auto_merge_applied")),
                retrieval_error=str(meta.get("retrieval_error") or ""),
            )
        )
    return results


def aggregate(
    results: list[EvalResult],
    cases: list[EvalCase],
    top_k: int,
    requested_retrieval_mode: str,
) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for block_type in sorted({result.block_type for result in results}):
        subset = [result for result in results if result.block_type == block_type]
        by_type[block_type] = {
            "cases": len(subset),
            f"hit_rate@{top_k}": mean(result.hit for result in subset),
            f"precision@{top_k}": mean(result.precision_at_k for result in subset),
            f"recall@{top_k}": mean(result.recall_at_k for result in subset),
            "mrr": mean(result.mrr for result in subset),
        }

    ragas_precision_values = [
        result.ragas_context_precision
        for result in results
        if result.ragas_context_precision is not None
    ]
    ragas_recall_values = [
        result.ragas_context_recall
        for result in results
        if result.ragas_context_recall is not None
    ]
    by_paper: dict[str, dict[str, Any]] = {}
    for paper_id in sorted({result.paper_id for result in results}):
        subset = [result for result in results if result.paper_id == paper_id]
        by_paper[paper_id] = {
            "cases": len(subset),
            f"hit_rate@{top_k}": mean(result.hit for result in subset),
            f"precision@{top_k}": mean(result.precision_at_k for result in subset),
            f"recall@{top_k}": mean(result.recall_at_k for result in subset),
            "mrr": mean(result.mrr for result in subset),
        }

    paper_ids = sorted({case.paper_id for case in cases})
    return {
        "paper_id": paper_ids[0] if len(paper_ids) == 1 else None,
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "case_count": len(results),
        "top_k": top_k,
        "requested_retrieval_mode": requested_retrieval_mode,
        f"hit_rate@{top_k}": mean(result.hit for result in results),
        f"precision@{top_k}": mean(result.precision_at_k for result in results),
        f"recall@{top_k}": mean(result.recall_at_k for result in results),
        "mrr": mean(result.mrr for result in results),
        "ragas_context_precision": mean(ragas_precision_values) if ragas_precision_values else None,
        "ragas_context_recall": mean(ragas_recall_values) if ragas_recall_values else None,
        "retrieval_modes": dict(Counter(result.retrieval_mode for result in results)),
        "rerank_applied_cases": sum(int(result.rerank_applied) for result in results),
        "auto_merge_applied_cases": sum(int(result.auto_merge_applied) for result in results),
        "by_paper": by_paper,
        "by_block_type": by_type,
    }


def write_outputs(
    output_root: Path,
    cases: list[EvalCase],
    results: list[EvalResult],
    summary: dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_label = summary["paper_id"] or f"multi_{summary['paper_count']}papers"
    run_dir = output_root / f"{timestamp}_{summary['requested_retrieval_mode']}_{paper_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "eval_dataset.jsonl").open("w", encoding="utf-8") as handle:
        for case, result in zip(cases, results, strict=True):
            payload = {
                **asdict(case),
                "retrieved_context_ids": result.retrieved_context_ids.split(";")
                if result.retrieved_context_ids
                else [],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with (run_dir / "per_case_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    return run_dir


def summary_markdown(summary: dict[str, Any]) -> str:
    top_k = summary["top_k"]
    paper_label = summary["paper_id"] or f"{summary['paper_count']} papers"
    lines = [
        "# Agentic RAG Retrieval Evaluation",
        "",
        f"- Paper Scope: `{paper_label}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Top K: `{top_k}`",
        f"- Requested Retrieval Mode: `{summary['requested_retrieval_mode']}`",
        f"- Hit Rate@K: `{summary[f'hit_rate@{top_k}']:.4f}`",
        f"- Precision@K: `{summary[f'precision@{top_k}']:.4f}`",
        f"- Recall@K: `{summary[f'recall@{top_k}']:.4f}`",
        f"- MRR: `{summary['mrr']:.4f}`",
        f"- RAGAS Context Precision: `{format_optional(summary['ragas_context_precision'])}`",
        f"- RAGAS Context Recall: `{format_optional(summary['ragas_context_recall'])}`",
        f"- Retrieval Modes: `{json.dumps(summary['retrieval_modes'], ensure_ascii=False)}`",
        f"- Rerank Applied Cases: `{summary['rerank_applied_cases']}`",
        f"- Auto Merge Applied Cases: `{summary['auto_merge_applied_cases']}`",
        "",
        "## By Paper",
        "",
        "| Paper ID | Cases | Hit Rate@K | Precision@K | Recall@K | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for paper_id, item in summary["by_paper"].items():
        lines.append(
            "| {paper_id} | {cases} | {hit:.4f} | {precision:.4f} | {recall:.4f} | {mrr:.4f} |".format(
                paper_id=paper_id,
                cases=item["cases"],
                hit=item[f"hit_rate@{top_k}"],
                precision=item[f"precision@{top_k}"],
                recall=item[f"recall@{top_k}"],
                mrr=item["mrr"],
            )
        )
    lines.extend(
        [
            "",
            "## By Block Type",
            "",
            "| Block Type | Cases | Hit Rate@K | Precision@K | Recall@K | MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for block_type, item in summary["by_block_type"].items():
        lines.append(
            "| {block_type} | {cases} | {hit:.4f} | {precision:.4f} | {recall:.4f} | {mrr:.4f} |".format(
                block_type=block_type,
                cases=item["cases"],
                hit=item[f"hit_rate@{top_k}"],
                precision=item[f"precision@{top_k}"],
                recall=item[f"recall@{top_k}"],
                mrr=item["mrr"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    if args.milvus_uri:
        settings.milvus_uri = args.milvus_uri
        vector_store._client = None

    if args.paper_id and args.all_papers:
        raise SystemExit("Use either --paper-id or --all-papers, not both.")

    paper_ids = indexed_paper_ids() if args.all_papers else [args.paper_id or latest_paper_id()]
    if not paper_ids:
        raise SystemExit(
            f"No indexed papers found under {settings.upload_dir}. Upload and index papers first."
        )

    block_types = {
        item.strip()
        for item in args.block_types.split(",")
        if item.strip()
    }
    cases: list[EvalCase] = []
    per_paper_limit = args.samples_per_paper or args.limit
    for paper_index, paper_id in enumerate(paper_ids, start=1):
        leaves = load_leaves(paper_id)
        cases.extend(
            build_cases(
                paper_id=paper_id,
                leaves=leaves,
                block_types=block_types,
                limit=per_paper_limit,
                seed=args.seed + paper_index,
                case_prefix=f"p{paper_index:02d}-" if len(paper_ids) > 1 else "",
            )
        )
    if not args.no_ragas:
        load_ragas()
    results = evaluate_cases(
        cases,
        top_k=args.top_k,
        use_ragas=not args.no_ragas,
        retrieval_mode=args.retrieval_mode,
        backend_url=args.backend_url,
    )
    summary = aggregate(
        results,
        cases,
        top_k=args.top_k,
        requested_retrieval_mode=args.retrieval_mode,
    )
    run_dir = write_outputs(Path(args.output_dir), cases, results, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote evaluation artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
