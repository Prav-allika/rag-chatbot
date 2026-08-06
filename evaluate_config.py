"""
Config Comparison Tool
Evaluate the CURRENT .env config against a fixed test set, so you can swap
models/settings, rerun, and compare numbers on equal footing.

Workflow:
    # 1. Once — generate and save a fixed question set from your document
    python evaluate_config.py --file data/resume.pdf --generate-testset --label "gpt-3.5-turbo baseline"

    # 2. Edit .env (new model / embedding / CONTEXTUAL_RETRIEVAL), then rerun
    #    against the SAME saved test set — no --generate-testset this time
    python evaluate_config.py --file data/resume.pdf --label "gpt-5-mini + contextual retrieval"

    # Optional: also score generation quality with RAGAS (needs RAGAS_EVAL=true in .env)
    python evaluate_config.py --file data/resume.pdf --ragas --label "gpt-5-mini + contextual retrieval"

Every run appends one line to artifacts/eval/results.jsonl so you can diff
runs across configs.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from app.config import Config
from app.rag_pipeline import build_vector_store, make_qa_chain
from app.evaluation import generate_retrieval_test_set, save_test_set, load_test_set, evaluate_rag_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TESTSET = "artifacts/eval/testset.json"
DEFAULT_RESULTS = "artifacts/eval/results.jsonl"


def _current_config() -> dict:
    """Snapshot of the config knobs that affect eval results, for the results log."""
    if Config.use_groq():
        llm = f"groq:{Config.GROQ_MODEL}"
    elif Config.use_openai():
        llm = f"openai:{Config.OPENAI_MODEL}"
    else:
        llm = f"local:{Config.LLM_MODEL}"
    embedding = Config.OPENAI_EMBEDDING_MODEL if Config.use_openai() else Config.EMBEDDING_MODEL
    return {
        "llm": llm,
        "embedding": embedding,
        "reranker": Config.RERANKER_MODEL,
        "contextual_retrieval": Config.CONTEXTUAL_RETRIEVAL,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="data/Attention.pdf", help="Document to evaluate against")
    parser.add_argument("--store", default="artifacts/vector_store/eval_run",
                         help="Vector store path — rebuilt fresh every run so it reflects the current config")
    parser.add_argument("--testset", default=DEFAULT_TESTSET, help="Path to the fixed test set JSON")
    parser.add_argument("--generate-testset", action="store_true",
                         help="(Re)generate the test set from this document and save it. "
                              "Do this ONCE per document, then omit it for every subsequent config run.")
    parser.add_argument("--n", type=int, default=8, help="Number of questions when generating a test set")
    parser.add_argument("--k", type=int, default=Config.RETRIEVAL_K, help="K for Precision@K / Recall@K")
    parser.add_argument("--ragas", action="store_true",
                         help="Also run RAGAS on a sample of questions (needs RAGAS_EVAL=true in .env)")
    parser.add_argument("--label", default="", help="Label for this run in the results log, e.g. 'gpt-5-mini'")
    args = parser.parse_args()

    logger.info(f"Building vector store for {args.file} under the CURRENT config...")
    vs, chunks = build_vector_store(args.file, args.store, doc_id="eval_run")
    chain = make_qa_chain(vs, doc_id="eval_run", all_chunks=chunks)

    if args.generate_testset or not os.path.exists(args.testset):
        logger.info(f"Generating a new {args.n}-question test set from this document...")
        test_set = generate_retrieval_test_set(chunks, args.n)
        if not test_set:
            logger.error("Could not generate a test set from this document — aborting.")
            sys.exit(1)
        save_test_set(test_set, args.testset)
    else:
        test_set = load_test_set(args.testset)
        logger.info(f"Reusing saved test set: {len(test_set)} question(s) from {args.testset}")

    retrieval_metrics = chain.run_retrieval_eval(k=args.k, test_set=test_set)

    ragas_avg = None
    if args.ragas:
        if not Config.RAGAS_EVAL:
            logger.warning("RAGAS_EVAL is not 'true' in .env — skipping RAGAS scoring.")
        else:
            logger.info("Running RAGAS on a sample of questions (calls your configured LLM as judge)...")
            scores_list = []
            for item in test_set[:5]:
                result = chain.invoke({"query": item["question"]})
                contexts = [s["content"] for s in result.get("sources", [])]
                if not contexts:
                    continue
                scores = evaluate_rag_response(item["question"], result["result"], contexts)
                if "error" not in scores and "info" not in scores:
                    scores_list.append(scores)
            if scores_list:
                keys = scores_list[0].keys()
                ragas_avg = {k: round(sum(s[k] for s in scores_list) / len(scores_list), 3) for k in keys}

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label or None,
        "config": _current_config(),
        "file": args.file,
        "retrieval_metrics": retrieval_metrics,
        "ragas_avg": ragas_avg,
    }
    os.makedirs(os.path.dirname(DEFAULT_RESULTS), exist_ok=True)
    with open(DEFAULT_RESULTS, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    logger.info("=" * 60)
    logger.info(f"Label:     {args.label or '(none)'}")
    logger.info(f"Config:    {run_record['config']}")
    logger.info(f"Retrieval: {retrieval_metrics}")
    if ragas_avg:
        logger.info(f"RAGAS:     {ragas_avg}")
    logger.info(f"Recorded -> {DEFAULT_RESULTS}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
