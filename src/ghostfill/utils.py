import hashlib
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

from .model import ClaimEvidence, FillRules, WritingStyleSheet


def ensure_nltk() -> None:
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def document_seed(base: int, *parts: str) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 15]


def word_count(text: str) -> int:
    return len(text.split())


def format_style_sheet(sheet: WritingStyleSheet) -> str:
    lines = ["=== AUTHOR WRITING STYLE SHEET ==="]
    dims = [
        ("plot", "PLOT (Structure & Flow)"),
        ("creativity", "CREATIVITY (Unique Elements)"),
        ("development", "DEVELOPMENT (Character & Setting)"),
        ("language_use", "LANGUAGE USE (Style & Voice)"),
    ]
    for attr, label in dims:
        pairs: List[ClaimEvidence] = getattr(sheet, attr)
        lines.append(f"\n[{label}]")
        for i, pair in enumerate(pairs, 1):
            lines.append(f"  {i}. Claim:    {pair.claim}")
            lines.append(f"     Evidence: {pair.evidence}")
    return "\n".join(lines)


def format_fill_rules(rules: FillRules) -> str:
    return (
        f"[PLOT]\n{rules.plot}\n\n"
        f"[CREATIVITY]\n{rules.creativity}\n\n"
        f"[DEVELOPMENT]\n{rules.development}\n\n"
        f"[LANGUAGE USE]\n{rules.language_use}"
    )
