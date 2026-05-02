"""
app/guards.py — Input Guard and Output Guard (PII redaction).

Input Guard  : blocks prompt injection / jailbreak attempts before retrieval.
Output Guard : redacts PII patterns from generated answers before display.

Both are stateless, regex-based, and have no dependency on the RAG chain.
"""

import re

# =============================================================================
# INPUT GUARD
# =============================================================================
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|all)\s+instructions",
    r"forget\s+(you are|your instructions|everything)",
    r"you are now\s+",
    r"\bact as( a)?\b",
    r"pretend (you|to be)",
    r"(jailbreak|developer mode|dan mode)",
    r"disregard\s+(previous|prior|all)",
    r"override\s+(your|all)\s+(instructions|guidelines)",
]


def check_input_guard(question: str) -> tuple:
    """
    Validate user input before it reaches the retrieval pipeline.
    Returns (is_safe: bool, reason: str).
    Runs in <1 ms — pure regex, no model calls.
    """
    q = question.strip()
    if len(q) < 2:
        return False, "Question is too short."
    if len(q) > 1500:
        return False, "Question exceeds maximum length (1500 chars)."
    q_lower = q.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Input contains potentially unsafe content and was blocked."
    return True, ""


# =============================================================================
# OUTPUT GUARD — PII Redaction
# =============================================================================
_PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",   "EMAIL"),
    (r"\b(\+\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b", "PHONE"),
    (r"\b\d{3}-\d{2}-\d{4}\b",                                  "SSN"),
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",               "CARD"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                            "IP_ADDR"),
    (r"\b(sk-|gsk_|sk-proj-)[A-Za-z0-9_-]{20,}\b",              "API_KEY"),
]


def redact_pii(text: str) -> tuple:
    """
    Scan generated answer text and replace PII with labelled placeholders.
    Returns (redacted_text: str, found_types: list[str]).
    """
    result = text
    found = []
    for pattern, label in _PII_PATTERNS:
        if re.search(pattern, result):
            result = re.sub(pattern, f"[{label}]", result)
            found.append(label)
    return result, found
