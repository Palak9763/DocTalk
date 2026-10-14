"""Phase 7: Comprehensive RAG Evaluation Suite runner.

Ablation evaluation across retrieval modes, hybrid fusion, reranking,
query rewriting, query expansion, HyDE, and Self-RAG.

CLI:
    python -m eval.run_eval --mode retrieval|full [--docs PATH] [--dataset PATH]
                            [--configs a,b,c] [--limit N] [--judge MODEL]
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

from tqdm import tqdm

from rag import config, llm
from rag.config import PRESETS, PipelineConfig
from rag.ingest import Chunk, load_chunks
from rag.pipeline import RAGPipeline
from eval import metrics, ragas_style

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Config Ablation Definitions
# ---------------------------------------------------------------------------

CONFIG_SPECS: list[tuple[str, PipelineConfig, str]] = [
    (
        "sparse",
        PipelineConfig(
            retrieval="sparse",
            use_rerank=False,
            use_rewrite=False,
            use_expansion=False,
            use_hyde=False,
            use_adaptive=False,
            use_grading=False,
            use_groundedness_check=False,
        ),
        "BM25 lexical search baseline (inverted index with Snowball stemming)",
    ),
    (
        "dense",
        PipelineConfig(
            retrieval="dense",
            use_rerank=False,
            use_rewrite=False,
            use_expansion=False,
            use_hyde=False,
            use_adaptive=False,
            use_grading=False,
            use_groundedness_check=False,
        ),
        "Bi-encoder dense cosine search baseline (nomic-embed-text)",
    ),
    (
        "hybrid",
        PipelineConfig(
            retrieval="hybrid",
            use_rerank=False,
            use_rewrite=False,
            use_expansion=False,
            use_hyde=False,
            use_adaptive=False,
            use_grading=False,
            use_groundedness_check=False,
        ),
        "+ Reciprocal Rank Fusion (RRF k=60) unifying sparse + dense candidates",
    ),
    (
        "hybrid_rerank",
        PipelineConfig(
            retrieval="hybrid",
            use_rerank=True,
            use_rewrite=False,
            use_expansion=False,
            use_hyde=False,
            use_adaptive=False,
            use_grading=False,
            use_groundedness_check=False,
        ),
        "+ Cross-encoder reranking (bge-reranker-v2-m3) over top-50 pool",
    ),
    (
        "+ rewrite",
        PipelineConfig(
            retrieval="hybrid",
            use_rerank=True,
            use_rewrite=True,
            use_expansion=False,
            use_hyde=False,
            use_adaptive=False,
            use_grading=False,
            use_groundedness_check=False,
        ),
        "+ Conversational coreference resolution on multi-turn dialogue",
    ),
    (
        "+ expand",
        PipelineConfig(
            retrieval="hybrid",
            use_rerank=True,
            use_rewrite=True,
            use_expansion=True,
            n_expansions=2,
            use_hyde=False,
            use_adaptive=False,
            use_grading=False,
            use_groundedness_check=False,
        ),
        "+ Multi-perspective query expansion (2 diverse variants)",
    ),
    (
        "+ hyde",
        PipelineConfig(
            retrieval="hybrid",
            use_rerank=True,
            use_rewrite=True,
            use_expansion=True,
            n_expansions=2,
            use_hyde=True,
            use_adaptive=False,
            use_grading=False,
            use_groundedness_check=False,
        ),
        "+ Hypothetical document embeddings (zero-shot alignment passage)",
    ),
    (
        "full_selfrag",
        copy.copy(PRESETS["thorough"]),
        "+ Self-RAG adaptive routing, chunk grading, and citation verification",
    ),
]

CONFIG_MAP: dict[str, tuple[PipelineConfig, str]] = {
    name: (cfg, desc) for name, cfg, desc in CONFIG_SPECS
}

ALIAS_MAP: dict[str, str] = {
    "sparse": "sparse",
    "dense": "dense",
    "hybrid": "hybrid",
    "hybrid_rerank": "hybrid_rerank",
    "rerank": "hybrid_rerank",
    "hybrid+rerank": "hybrid_rerank",
    "+ rewrite": "+ rewrite",
    "+rewrite": "+ rewrite",
    "rewrite": "+ rewrite",
    "+ expand": "+ expand",
    "+expand": "+ expand",
    "expand": "+ expand",
    "+ hyde": "+ hyde",
    "+hyde": "+ hyde",
    "hyde": "+ hyde",
    "full_selfrag": "full_selfrag",
    "selfrag": "full_selfrag",
    "thorough": "full_selfrag",
}


def resolve_gold_chunk_ids(
    gold_entries: list[list[str]], chunks: list[Chunk]
) -> set[str]:
    """Resolve [[source_file, section_heading], ...] to chunk IDs in chunks."""
    matched: set[str] = set()
    for entry in gold_entries:
        if len(entry) < 2:
            continue
        src, sec = entry[0], entry[1]
        for c in chunks:
            if c.source == src and c.section == sec:
                matched.add(c.id)
    return matched


def count_llm_calls(cfg: PipelineConfig, trace: dict[str, Any], has_history: bool) -> int:
    """Calculate the estimated number of LLM invocations for a single pipeline run."""
    timings = trace.get("timings_ms", {})
    calls = 0
    if cfg.use_adaptive or "decide" in timings:
        calls += 1
    if (cfg.use_rewrite and has_history) or "rewrite" in timings:
        calls += 1
    if cfg.use_expansion or "expand" in timings:
        calls += 1
    if cfg.use_hyde or "hyde" in timings:
        calls += 1
    if cfg.use_grading and cfg.grading_mode == "llm":
        calls += len(trace.get("retrieved", []))
    if "generate" in timings:
        calls += 1
    if cfg.use_groundedness_check or "groundedness" in timings:
        calls += 1
    return max(1, calls)


# ---------------------------------------------------------------------------
# Runner Execution
# ---------------------------------------------------------------------------


def run_evaluation(
    mode: str,
    docs_path: Path | str,
    dataset_path: Path | str,
    selected_configs: list[str] | None = None,
    limit: int | None = None,
    judge_model: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Execute evaluation and return (raw_data_dict, markdown_report)."""
    docs_dir = Path(docs_path)
    ds_file = Path(dataset_path)

    if not ds_file.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {ds_file}")

    dataset: list[dict[str, Any]] = json.loads(ds_file.read_text(encoding="utf-8"))
    if limit is not None and limit > 0:
        dataset = dataset[:limit]

    logger.info("Loading document chunks from %s", docs_dir)
    chunks = load_chunks(docs_dir)
    if not chunks:
        raise ValueError(f"No chunks found in documents directory {docs_dir}")

    # Build one base pipeline and reuse indices & reranker across configs
    logger.info("Building base RAGPipeline over %d chunks…", len(chunks))
    base_pipeline = RAGPipeline(cfg=PRESETS["thorough"], chunks=chunks)

    # Determine configs to evaluate
    if selected_configs:
        config_names: list[str] = []
        for c in selected_configs:
            clean = c.strip().lower()
            canonical = ALIAS_MAP.get(clean, clean)
            if canonical in CONFIG_MAP:
                config_names.append(canonical)
            else:
                logger.warning("Unrecognized config '%s', skipping", c)
    else:
        config_names = [name for name, _, _ in CONFIG_SPECS]

    if not config_names:
        raise ValueError("No valid evaluation configurations selected")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_per_config: dict[str, list[dict[str, Any]]] = {}
    config_summaries: dict[str, dict[str, Any]] = {}
    type_summaries_per_config: dict[str, dict[str, dict[str, Any]]] = {}

    ragas_style.reset_judge_stats()

    print(f"\n[EVAL] Running Phase 7 Evaluation Suite [Mode: {mode.upper()}]")
    print(f"Corpus Chunks: {len(chunks)} | Questions: {len(dataset)} | Configs: {len(config_names)}")
    if mode == "full":
        print(f"Judge Model: {judge_model or config.JUDGE_MODEL}")
    print("-" * 70)

    for cfg_name in config_names:
        cfg, desc = CONFIG_MAP[cfg_name]
        pipeline = base_pipeline.with_config(cfg)
        cfg_results: list[dict[str, Any]] = []

        print(f"\n>> Evaluating Config: '{cfg_name}' ({pipeline.cfg.label()})")

        pbar = tqdm(dataset, desc=f"{cfg_name:<14}", unit="q", file=sys.stdout)
        for item in pbar:
            qid = item["id"]
            question = item["question"]
            history = item.get("history") or []
            gold = item.get("gold") or []
            reference = item.get("reference") or ""
            qtype = item.get("type", "factual")
            needs_retrieval = item.get("needs_retrieval", True)

            gold_ids = resolve_gold_chunk_ids(gold, chunks)

            res_entry: dict[str, Any] = {
                "id": qid,
                "type": qtype,
                "question": question,
                "needs_retrieval": needs_retrieval,
                "gold_ids": sorted(list(gold_ids)),
            }

            try:
                if mode == "retrieval":
                    ranked_indices, trace = pipeline.retrieve(
                        question, history=history
                    )
                    ranked_ids = [chunks[i].id for i in ranked_indices]
                    res_entry["ranked_ids"] = ranked_ids

                    if gold_ids:
                        rep = metrics.retrieval_report(ranked_ids, gold_ids, k=5)
                        res_entry.update(rep)
                    else:
                        res_entry["_skip"] = True
                        res_entry["is_empty_gold"] = True

                else:  # mode == "full"
                    t0 = time.perf_counter()
                    ans_result = pipeline.answer(question, history=history)
                    latency = (time.perf_counter() - t0) * 1000.0

                    trace = ans_result.trace
                    retrieved_ids = trace.get("retrieved") or []
                    res_entry["ranked_ids"] = retrieved_ids
                    res_entry["answer"] = ans_result.answer
                    res_entry["used_retrieval"] = ans_result.used_retrieval
                    res_entry["latency_ms"] = trace.get("timings_ms", {}).get(
                        "total", round(latency, 1)
                    )
                    res_entry["llm_calls"] = count_llm_calls(
                        pipeline.cfg, trace, bool(history)
                    )

                    # Retrieval metrics if gold available
                    if gold_ids:
                        rep = metrics.retrieval_report(retrieved_ids, gold_ids, k=5)
                        res_entry.update(rep)
                    else:
                        res_entry["is_empty_gold"] = True

                    # Self-RAG Decision accuracy
                    res_entry["decision_correct"] = (
                        1.0 if ans_result.used_retrieval == needs_retrieval else 0.0
                    )
                    # Skipping accuracy: did it skip retrieval when not needed?
                    if not needs_retrieval:
                        res_entry["skip_correct"] = (
                            1.0 if not ans_result.used_retrieval else 0.0
                        )

                    # Groundedness rate
                    res_entry["grounded"] = (
                        1.0 if trace.get("grounded") is True else 0.0
                    )

                    # RAGAS-style judge metrics
                    ctx_texts = [c.text for c in ans_result.contexts]
                    res_entry["context_precision"] = (
                        ragas_style.context_precision(
                            question, ctx_texts, judge_model=judge_model
                        )
                    )
                    res_entry["context_recall"] = (
                        ragas_style.context_recall(
                            question, reference, ctx_texts, judge_model=judge_model
                        )
                    )
                    res_entry["faithfulness"] = (
                        ragas_style.faithfulness(
                            question,
                            ans_result.answer,
                            ctx_texts,
                            judge_model=judge_model,
                        )
                    )
                    res_entry["answer_relevancy"] = (
                        ragas_style.answer_relevancy(
                            question, ans_result.answer, judge_model=judge_model
                        )
                    )

            except Exception as exc:
                logger.error("Error evaluating question %s on %s: %s", qid, cfg_name, exc)
                res_entry["error"] = str(exc)
                res_entry["_skip"] = True

            cfg_results.append(res_entry)

        results_per_config[cfg_name] = cfg_results

        # Aggregate overall
        agg = metrics.aggregate(cfg_results)
        # Extra aggregations for full mode
        if mode == "full":
            dec_scores = [
                r["decision_correct"]
                for r in cfg_results
                if "decision_correct" in r and not r.get("error")
            ]
            agg["decision_accuracy"] = (
                sum(dec_scores) / len(dec_scores) if dec_scores else 0.0
            )

            skip_scores = [
                r["skip_correct"]
                for r in cfg_results
                if "skip_correct" in r and not r.get("error")
            ]
            agg["skip_accuracy"] = (
                sum(skip_scores) / len(skip_scores) if skip_scores else 1.0
            )

            ground_scores = [
                r["grounded"]
                for r in cfg_results
                if "grounded" in r and not r.get("error")
            ]
            agg["groundedness_rate"] = (
                sum(ground_scores) / len(ground_scores) if ground_scores else 0.0
            )

            lats = [
                r["latency_ms"]
                for r in cfg_results
                if "latency_ms" in r and not r.get("error")
            ]
            agg["avg_latency_ms"] = sum(lats) / len(lats) if lats else 0.0

            calls = [
                r["llm_calls"]
                for r in cfg_results
                if "llm_calls" in r and not r.get("error")
            ]
            agg["avg_llm_calls"] = sum(calls) / len(calls) if calls else 0.0

            for ragas_k in (
                "context_precision",
                "context_recall",
                "faithfulness",
                "answer_relevancy",
            ):
                vals = [
                    r[ragas_k]
                    for r in cfg_results
                    if r.get(ragas_k) is not None and isinstance(r.get(ragas_k), (int, float))
                ]
                agg[ragas_k] = sum(vals) / len(vals) if vals else 0.0

        config_summaries[cfg_name] = agg

        # Aggregate per question type
        by_type: dict[str, list[dict[str, Any]]] = {}
        for r in cfg_results:
            by_type.setdefault(r["type"], []).append(r)

        type_aggs: dict[str, dict[str, Any]] = {}
        for tname, treps in by_type.items():
            type_aggs[tname] = metrics.aggregate(treps)
        type_summaries_per_config[cfg_name] = type_aggs

    judge_stats = ragas_style.get_judge_stats()

    # Build Markdown and Terminal reports
    report_md = _build_markdown_report(
        mode=mode,
        timestamp=timestamp,
        chunks_count=len(chunks),
        questions_count=len(dataset),
        judge_model=judge_model or config.JUDGE_MODEL,
        config_names=config_names,
        config_summaries=config_summaries,
        type_summaries=type_summaries_per_config,
        judge_stats=judge_stats,
    )

    # Save to eval/results/
    out_dir = config.ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_file = out_dir / f"{timestamp}_{mode}.json"
    md_file = out_dir / f"{timestamp}_{mode}.md"

    raw_output = {
        "metadata": {
            "mode": mode,
            "timestamp": timestamp,
            "docs_path": str(docs_path),
            "dataset_path": str(dataset_path),
            "chunks_count": len(chunks),
            "questions_count": len(dataset),
            "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL,
            "rerank_model": config.RERANK_MODEL,
            "judge_model": judge_model or config.JUDGE_MODEL,
            "judge_stats": judge_stats,
        },
        "config_summaries": config_summaries,
        "type_summaries": type_summaries_per_config,
        "raw_results": results_per_config,
    }

    json_file.write_text(json.dumps(raw_output, indent=2), encoding="utf-8")
    md_file.write_text(report_md, encoding="utf-8")

    logger.info("Saved raw results to %s", json_file)
    logger.info("Saved report to %s", md_file)

    # Print report to stdout
    print("\n" + report_md + "\n")

    return raw_output, report_md


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------


def _bold_max(val: float, max_val: float, fmt: str = ".3f") -> str:
    """Format float and bold if within 0.001 of maximum value."""
    s = f"{val:{fmt}}"
    if max_val > 0.0 and abs(val - max_val) < 0.001:
        return f"**{s}**"
    return s


def _build_markdown_report(
    mode: str,
    timestamp: str,
    chunks_count: int,
    questions_count: int,
    judge_model: str,
    config_names: list[str],
    config_summaries: dict[str, dict[str, Any]],
    type_summaries: dict[str, dict[str, dict[str, Any]]],
    judge_stats: dict[str, int],
) -> str:
    """Generate human-readable GitHub Flavored Markdown evaluation report."""
    lines: list[str] = []

    lines.append(f"# RAG Evaluation Suite Report ({mode.upper()})")
    lines.append(f"**Date:** {timestamp}  ")
    lines.append(f"**Corpus Chunks:** {chunks_count} | **Questions Evaluated:** {questions_count}  ")
    lines.append(
        f"**Models:** LLM `{config.CHAT_MODEL}` | Embed `{config.EMBED_MODEL}` | Reranker `{config.RERANK_MODEL}`  "
    )
    if mode == "full":
        lines.append(
            f"**Judge Model:** `{judge_model}` (Total Calls: {judge_stats['calls']}, Skips: {judge_stats['skips']})  "
        )

    lines.append("")
    lines.append(
        "> ⚠️ **Warning**: Small local judge models (e.g. 3B parameters) give noisy RAGAS-style scores and the evaluation dataset is compact (~20 items), so only large differences in metrics should be considered statistically meaningful."
    )
    lines.append("")

    # Main Benchmark Table
    lines.append("## 1. Cumulative Ablation Benchmark")
    lines.append("")

    if mode == "retrieval":
        # Determine column maximums for bolding
        max_hit = max(
            (config_summaries[c].get("hit@5", 0.0) for c in config_names), default=0.0
        )
        max_rec = max(
            (config_summaries[c].get("recall@5", 0.0) for c in config_names),
            default=0.0,
        )
        max_prec = max(
            (config_summaries[c].get("precision@5", 0.0) for c in config_names),
            default=0.0,
        )
        max_mrr = max(
            (config_summaries[c].get("mrr", 0.0) for c in config_names), default=0.0
        )
        max_ndcg = max(
            (config_summaries[c].get("ndcg@5", 0.0) for c in config_names), default=0.0
        )

        lines.append(
            "| Config | Pipeline Toggles | Hit@5 | Recall@5 | Prec@5 | MRR | nDCG@5 | Evaluated |"
        )
        lines.append(
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        for name in config_names:
            s = config_summaries[name]
            cfg, _ = CONFIG_MAP[name]
            hit = _bold_max(s.get("hit@5", 0.0), max_hit)
            rec = _bold_max(s.get("recall@5", 0.0), max_rec)
            prc = _bold_max(s.get("precision@5", 0.0), max_prec)
            mr = _bold_max(s.get("mrr", 0.0), max_mrr)
            nd = _bold_max(s.get("ndcg@5", 0.0), max_ndcg)
            cnt = s.get("count", 0)

            lines.append(
                f"| `{name}` | `{cfg.label()}` | {hit} | {rec} | {prc} | {mr} | {nd} | {cnt} |"
            )

    else:  # mode == "full"
        max_hit = max(
            (config_summaries[c].get("hit@5", 0.0) for c in config_names), default=0.0
        )
        max_rec = max(
            (config_summaries[c].get("recall@5", 0.0) for c in config_names),
            default=0.0,
        )
        max_mrr = max(
            (config_summaries[c].get("mrr", 0.0) for c in config_names), default=0.0
        )
        max_cprec = max(
            (config_summaries[c].get("context_precision", 0.0) for c in config_names),
            default=0.0,
        )
        max_crec = max(
            (config_summaries[c].get("context_recall", 0.0) for c in config_names),
            default=0.0,
        )
        max_faith = max(
            (config_summaries[c].get("faithfulness", 0.0) for c in config_names),
            default=0.0,
        )
        max_rel = max(
            (config_summaries[c].get("answer_relevancy", 0.0) for c in config_names),
            default=0.0,
        )
        max_acc = max(
            (config_summaries[c].get("decision_accuracy", 0.0) for c in config_names),
            default=0.0,
        )

        lines.append(
            "| Config | Hit@5 | Recall@5 | MRR | C-Prec | C-Rec | Faith | Relevancy | Skip Acc | Latency | Calls |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        for name in config_names:
            s = config_summaries[name]
            hit = _bold_max(s.get("hit@5", 0.0), max_hit)
            rec = _bold_max(s.get("recall@5", 0.0), max_rec)
            mr = _bold_max(s.get("mrr", 0.0), max_mrr)
            cp = _bold_max(s.get("context_precision", 0.0), max_cprec)
            cr = _bold_max(s.get("context_recall", 0.0), max_crec)
            ft = _bold_max(s.get("faithfulness", 0.0), max_faith)
            ar = _bold_max(s.get("answer_relevancy", 0.0), max_rel)
            sa = f"{s.get('skip_accuracy', 1.0):.2f}"
            lat = f"{s.get('avg_latency_ms', 0.0):.0f} ms"
            calls = f"{s.get('avg_llm_calls', 0.0):.1f}"

            lines.append(
                f"| `{name}` | {hit} | {rec} | {mr} | {cp} | {cr} | {ft} | {ar} | {sa} | {lat} | {calls} |"
            )

    lines.append("")
    lines.append("### Marginal Technique Contributions")
    for name in config_names:
        _, desc = CONFIG_MAP[name]
        lines.append(f"- **`{name}`**: {desc}")
    lines.append("")

    # Per Question Type Breakdown
    lines.append("## 2. Retrieval Performance by Question Type")
    lines.append("")

    types_seen: list[str] = []
    for c in config_names:
        for t in type_summaries.get(c, {}):
            if t not in types_seen:
                types_seen.append(t)

    lines.append("| Question Type | Config | Hit@5 | Recall@5 | Prec@5 | MRR | nDCG@5 | Count |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for qtype in sorted(types_seen):
        for cname in config_names:
            tinfo = type_summaries.get(cname, {}).get(qtype)
            if not tinfo or tinfo.get("count", 0) == 0:
                continue
            h = f"{tinfo.get('hit@5', 0.0):.3f}"
            r = f"{tinfo.get('recall@5', 0.0):.3f}"
            p = f"{tinfo.get('precision@5', 0.0):.3f}"
            m = f"{tinfo.get('mrr', 0.0):.3f}"
            n = f"{tinfo.get('ndcg@5', 0.0):.3f}"
            cnt = tinfo.get("count", 0)
            lines.append(
                f"| `{qtype}` | `{cname}` | {h} | {r} | {p} | {m} | {n} | {cnt} |"
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI parser and runner dispatch."""
    parser = argparse.ArgumentParser(
        description="Phase 7: Chat With Your Docs Evaluation Suite"
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        required=True,
        help="Evaluation mode: 'retrieval' (fast, no answers) or 'full' (with answers & RAGAS)",
    )
    parser.add_argument(
        "--docs",
        default=str(config.NOTES_DIR),
        help="Path to documents folder (default: data/notes)",
    )
    parser.add_argument(
        "--dataset",
        default=str(config.ROOT / "eval" / "dataset.json"),
        help="Path to evaluation dataset JSON (default: eval/dataset.json)",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated configs to evaluate (e.g. 'sparse,dense,hybrid,hybrid_rerank')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions evaluated for quick testing",
    )
    parser.add_argument(
        "--judge",
        default=None,
        help=f"Judge LLM model name (default: {config.JUDGE_MODEL})",
    )

    args = parser.parse_args()

    selected = [c.strip() for c in args.configs.split(",")] if args.configs else None

    run_evaluation(
        mode=args.mode,
        docs_path=args.docs,
        dataset_path=args.dataset,
        selected_configs=selected,
        limit=args.limit,
        judge_model=args.judge,
    )


if __name__ == "__main__":
    main()
