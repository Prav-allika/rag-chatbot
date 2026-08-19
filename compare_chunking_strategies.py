"""
Chunking Strategy Comparison Tool
Runs the same fixed golden test set against all three chunking strategies
(fixed, structure, semantic) on the same document, using the same
embedding/LLM config, and reports retrieval metrics + chunk stats per
strategy — so you pick a chunker with numbers instead of a guess.

Workflow:
    # 1. Once — generate and save a fixed question set from your document
    python compare_chunking_strategies.py --file data/Attention.pdf --generate-testset --n 12

    # 2. Rerun any time (e.g. after changing CHUNK_SIZE) against the SAME
    #    saved test set for an apples-to-apples comparison
    python compare_chunking_strategies.py --file data/Attention.pdf

    # Or use the hand-curated, strategy-neutral golden set instead (see
    # artifacts/eval/golden_testset.json / run_golden_eval.py) — avoids the
    # bias of a synthetic set generated from one strategy's own chunks
    python compare_chunking_strategies.py --file data/Attention.pdf --golden

Every run appends one record to artifacts/eval/chunking_comparison.jsonl and
(re)writes a markdown report to artifacts/eval/chunking_comparison.md.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from app.config import Config
from app.rag_pipeline import build_vector_store, make_qa_chain
from app.evaluation import generate_retrieval_test_set, save_test_set, load_test_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES = ["fixed", "structure", "semantic"]
DEFAULT_TESTSET = "artifacts/eval/testset.json"
DEFAULT_RESULTS = "artifacts/eval/chunking_comparison.jsonl"
DEFAULT_REPORT = "artifacts/eval/chunking_comparison.md"

METRIC_KEYS = ["precision_at_k", "recall_at_k", "mrr", "coverage"]


def _run_strategy(file_path: str, strategy: str, test_set: list, k: int) -> dict:
    Config.CHUNKING_STRATEGY = strategy
    store_path = f"artifacts/vector_store/chunkeval_{strategy}"
    doc_id = f"chunkeval_{strategy}"

    start = time.monotonic()
    vs, chunks = build_vector_store(file_path, store_path, doc_id=doc_id)
    build_seconds = round(time.monotonic() - start, 1)

    chain = make_qa_chain(vs, doc_id=doc_id, all_chunks=chunks)
    metrics = chain.run_retrieval_eval(k=k, test_set=test_set)

    char_counts = [len(c.page_content) for c in chunks] or [0]
    return {
        "strategy": strategy,
        "num_chunks": len(chunks),
        "avg_chars": round(sum(char_counts) / len(char_counts), 1),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "build_seconds": build_seconds,
        **metrics,
    }


def _write_markdown_report(file_path: str, k: int, results: list, path: str) -> None:
    lines = [
        "# Chunking Strategy Comparison",
        "",
        f"Document: `{file_path}`  ·  k={k}  ·  generated {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Strategy | Precision@K | Recall@K | MRR | Coverage | # Chunks | Avg chars | Build time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['strategy']} | {r.get('precision_at_k', 'n/a')} | {r.get('recall_at_k', 'n/a')} | "
            f"{r.get('mrr', 'n/a')} | {r.get('coverage', 'n/a')} | {r['num_chunks']} | "
            f"{r['avg_chars']} | {r['build_seconds']} |"
        )

    lines.append("")
    for metric in METRIC_KEYS:
        scored = [(r["strategy"], r[metric]) for r in results if metric in r]
        if scored:
            best = max(scored, key=lambda x: x[1])
            lines.append(f"- **{metric}** winner: `{best[0]}` ({best[1]})")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Markdown report written -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="data/Attention.pdf", help="Document to evaluate against")
    parser.add_argument("--testset", default=DEFAULT_TESTSET, help="Path to the fixed test set JSON")
    parser.add_argument("--generate-testset", action="store_true",
                         help="(Re)generate the test set from this document (fixed-strategy chunks) and save it. "
                              "Do this ONCE per document, then omit it for subsequent runs.")
    parser.add_argument("--n", type=int, default=12, help="Number of questions when generating a test set")
    parser.add_argument("--k", type=int, default=Config.RETRIEVAL_K, help="K for Precision@K / Recall@K")
    parser.add_argument("--golden", action="store_true",
                         help="Use artifacts/eval/golden_testset.json (lookup + multi_hop questions only) "
                              "instead of an LLM-synthetic set. Strategy-neutral: the synthetic set is "
                              "generated FROM one strategy's chunks, which structurally favors that "
                              "strategy in the comparison — the golden set's expected_answer text is "
                              "written independently of any chunking strategy, so word-overlap against "
                              "it is a fair comparison across all three.")
    args = parser.parse_args()

    original_strategy = Config.CHUNKING_STRATEGY

    if args.golden:
        with open("artifacts/eval/golden_testset.json") as f:
            golden = json.load(f)
        test_set = [
            {"question": q["question"], "source_content": q["expected_answer"]}
            for q in golden if q.get("expected_answer")
        ]
        logger.info(f"Using golden test set: {len(test_set)} answerable question(s) (lookup + multi_hop)")
    elif args.generate_testset or not os.path.exists(args.testset):
        logger.info(f"Generating a new {args.n}-question test set from this document (fixed-strategy chunks)...")
        Config.CHUNKING_STRATEGY = "fixed"
        _, seed_chunks = build_vector_store(
            args.file, "artifacts/vector_store/chunkeval_seed", doc_id="chunkeval_seed"
        )
        test_set = generate_retrieval_test_set(seed_chunks, args.n)
        if not test_set:
            logger.error("Could not generate a test set from this document — aborting.")
            sys.exit(1)
        save_test_set(test_set, args.testset)
    else:
        test_set = load_test_set(args.testset)
        logger.info(f"Reusing saved test set: {len(test_set)} question(s) from {args.testset}")

    results = []
    for strategy in STRATEGIES:
        logger.info(f"--- Building + evaluating strategy: {strategy} ---")
        results.append(_run_strategy(args.file, strategy, test_set, args.k))

    Config.CHUNKING_STRATEGY = original_strategy

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": args.file,
        "k": args.k,
        "n_questions": len(test_set),
        "results": results,
    }
    os.makedirs(os.path.dirname(DEFAULT_RESULTS), exist_ok=True)
    with open(DEFAULT_RESULTS, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    _write_markdown_report(args.file, args.k, results, DEFAULT_REPORT)

    logger.info("=" * 70)
    for r in results:
        logger.info(f"{r['strategy']:<10} {r}")
    logger.info(f"Recorded -> {DEFAULT_RESULTS}")
    logger.info(f"Report   -> {DEFAULT_REPORT}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
