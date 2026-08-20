"""
app/golden_eval_summary.py — parses the golden eval results file into a
per-category correctness summary.

Shared by app.py and streamlit_app.py's "Golden Set" view so both read the
same run the same way.
"""

import json
import os

GOLDEN_RESULTS_PATH = "artifacts/eval/golden_eval_results.jsonl"


def load_golden_eval_summary(path: str = GOLDEN_RESULTS_PATH) -> dict:
    """Reads the most recent full (50-question) golden eval run and returns
    a per-category correctness breakdown, or None if no full run is recorded."""
    if not os.path.exists(path):
        return None
    latest_full_run = None
    with open(path) as f:
        for line in f:
            run = json.loads(line)
            if run.get("n_questions") == 50:
                latest_full_run = run
    if latest_full_run is None:
        return None

    by_category = {}
    for r in latest_full_run["results"]:
        cat = r["category"]
        by_category.setdefault(cat, [0, 0])
        by_category[cat][1] += 1
        if r["correct"]:
            by_category[cat][0] += 1

    total_correct = sum(c for c, _ in by_category.values())
    total_n = sum(n for _, n in by_category.values())
    return {
        "timestamp": latest_full_run["timestamp"],
        "by_category": by_category,
        "overall": (total_correct, total_n),
    }
