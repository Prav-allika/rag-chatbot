"""
Golden Eval Suite
Runs the hand-curated 50-question golden set (artifacts/eval/golden_testset.json)
through the full QA chain and scores each answer on:
  - retrieval confidence   (free, from the composite confidence score — gap #5)
  - citation accuracy      (LLM-as-judge per cited sentence — gap #4, app/citations.py)
  - answer correctness     (LLM-as-judge vs the golden expected_answer; for the
                             no_answer category, correctness means "did it refuse";
                             for ambiguous, "did it handle the ambiguity gracefully")
  - faithfulness           (RAGAS, opt-in via --ragas, same RAGAS_EVAL gate as elsewhere)

Costs real LLM calls: one invoke per question (a few calls each — condense/decompose/
route/generate), one citation-verification call per cited sentence, and one
correctness-judge call per question. For a full 50-question run that adds up —
use --n to smoke-test on a small subset first.

Usage:
    python run_golden_eval.py --n 5              # cheap smoke test
    python run_golden_eval.py                    # full 50-question run
    python run_golden_eval.py --ragas             # + RAGAS faithfulness (needs RAGAS_EVAL=true)
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

from app.config import Config
from app.rag_pipeline import build_vector_store, make_qa_chain, get_llm, evaluate_rag_response
from app.citations import verify_citations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TESTSET = "artifacts/eval/golden_testset.json"
DEFAULT_RESULTS = "artifacts/eval/golden_eval_results.jsonl"
DEFAULT_REPORT = "artifacts/eval/golden_eval_summary.md"

_ANSWERABLE_PROMPT = """You are grading a RAG system's answer against a reference answer.

Question: {question}
Reference answer: {expected}
System's answer: {answer}

Does the system's answer correctly convey the key facts in the reference answer? It
does not need to match wording exactly. Reply with exactly one line:
CORRECT: <one short reason>
or
INCORRECT: <one short reason>"""

_NO_ANSWER_PROMPT = """The following question has no answer in the source document, so the
correct behavior is to decline rather than guess.

Question: {question}
System's answer: {answer}

Did the system correctly decline to answer, rather than fabricating a confident but
ungrounded answer? Reply with exactly one line:
CORRECT: <one short reason>
or
INCORRECT: <one short reason>"""

_AMBIGUOUS_PROMPT = """The following question is ambiguous or underspecified.

Question: {question}
System's answer: {answer}

Did the system handle the ambiguity gracefully -- e.g. by declining, asking for
clarification, or noting multiple interpretations -- rather than confidently guessing
one specific interpretation? Reply with exactly one line:
CORRECT: <one short reason>
or
INCORRECT: <one short reason>"""


def _judge(question: str, answer: str, expected: str, category: str) -> dict:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if category == "no_answer":
        template, inputs = _NO_ANSWER_PROMPT, {"question": question, "answer": answer}
    elif category == "ambiguous":
        template, inputs = _AMBIGUOUS_PROMPT, {"question": question, "answer": answer}
    else:
        template, inputs = _ANSWERABLE_PROMPT, {"question": question, "expected": expected, "answer": answer}

    prompt = PromptTemplate(template=template, input_variables=list(inputs.keys()))
    chain = prompt | get_llm() | StrOutputParser()
    try:
        raw = chain.invoke(inputs).strip()
        correct = raw.upper().startswith("CORRECT")
        reason = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return {"correct": correct, "reason": reason}
    except Exception as e:
        logger.warning(f"Correctness judge failed: {e}")
        return {"correct": False, "reason": f"Judge call failed: {e}"}


def _write_markdown_report(results: list, path: str) -> None:
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    lines = [
        "# Golden Eval Summary",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}  ·  {len(results)} question(s)",
        "",
        "| Category | N | Correctness | Avg Retrieval Conf | Avg Citation Accuracy |",
        "|---|---|---|---|---|",
    ]
    for cat, items in sorted(by_cat.items()):
        n = len(items)
        correctness = sum(i["correct"] for i in items) / n
        rc = [i["retrieval_confidence"] for i in items if i["retrieval_confidence"] is not None]
        ca = [i["citation_accuracy"] for i in items if i["citation_accuracy"] is not None]
        avg_rc = f"{sum(rc)/len(rc):.3f}" if rc else "n/a"
        avg_ca = f"{sum(ca)/len(ca):.3f}" if ca else "n/a"
        lines.append(f"| {cat} | {n} | {correctness:.3f} | {avg_rc} | {avg_ca} |")

    overall_correct = sum(r["correct"] for r in results) / len(results)
    lines += ["", f"**Overall correctness: {overall_correct:.3f}** ({sum(r['correct'] for r in results)}/{len(results)})"]

    incorrect = [r for r in results if not r["correct"]]
    if incorrect:
        lines += ["", f"## Incorrect ({len(incorrect)})", ""]
        for r in incorrect[:15]:
            lines.append(f"- **[{r['category']}]** \"{r['question']}\" -> {r['correctness_reason']}")
        if len(incorrect) > 15:
            lines.append(f"- ... and {len(incorrect) - 15} more")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Report written -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="data/Attention.pdf")
    parser.add_argument("--store", default="artifacts/vector_store/golden_eval")
    parser.add_argument("--testset", default=DEFAULT_TESTSET)
    parser.add_argument("--n", type=int, default=None, help="Only run the first N questions (cheap smoke test)")
    parser.add_argument("--category", default=None,
                         help="Only run questions in this category (lookup | multi_hop | no_answer | ambiguous) "
                              "— cheap way to re-test one category after a targeted fix.")
    parser.add_argument("--ragas", action="store_true", help="Also score faithfulness via RAGAS (needs RAGAS_EVAL=true)")
    args = parser.parse_args()

    with open(args.testset) as f:
        test_set = json.load(f)
    if args.category:
        test_set = [q for q in test_set if q["category"] == args.category]
    if args.n:
        test_set = test_set[: args.n]
    logger.info(f"Running golden eval: {len(test_set)} question(s) from {args.testset}")

    vs, chunks = build_vector_store(args.file, args.store, doc_id="golden_eval")
    chain = make_qa_chain(vs, doc_id="golden_eval", all_chunks=chunks)

    results = []
    for i, item in enumerate(test_set, 1):
        question = item["question"]
        expected = item.get("expected_answer")
        category = item["category"]
        logger.info(f"[{i}/{len(test_set)}] ({category}) {question[:70]}")

        try:
            result = chain.invoke({"query": question})
        except Exception as e:
            logger.warning(f"invoke() failed: {e}")
            results.append({
                "question": question, "category": category, "answer": f"ERROR: {e}",
                "grade": "ERROR", "retrieval_confidence": None, "citation_coverage": None,
                "citation_accuracy": None, "correct": False, "correctness_reason": "invoke() failed",
                "ragas": None,
            })
            continue

        answer = result.get("result", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", {})

        citation_result = verify_citations(answer, sources) if sources else {}
        correctness = _judge(question, answer, expected, category)

        ragas_scores = None
        if args.ragas and Config.RAGAS_EVAL and sources:
            contexts = [s.get("full_content", s.get("content", "")) for s in sources]
            ragas_scores = evaluate_rag_response(question, answer, contexts)

        results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "grade": result.get("grade"),
            "retrieval_confidence": confidence.get("retrieval"),
            "citation_coverage": confidence.get("citation_coverage"),
            "citation_accuracy": citation_result.get("accuracy"),
            "correct": correctness["correct"],
            "correctness_reason": correctness["reason"],
            "ragas": ragas_scores,
        })

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": args.file,
        "n_questions": len(results),
        "results": results,
    }
    os.makedirs(os.path.dirname(DEFAULT_RESULTS), exist_ok=True)
    with open(DEFAULT_RESULTS, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    _write_markdown_report(results, DEFAULT_REPORT)

    overall_correct = sum(r["correct"] for r in results) / len(results) if results else 0.0
    logger.info("=" * 60)
    logger.info(f"Overall correctness: {overall_correct:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
    logger.info(f"Recorded -> {DEFAULT_RESULTS}")
    logger.info(f"Report   -> {DEFAULT_REPORT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
