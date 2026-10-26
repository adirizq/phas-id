import difflib
import random
from typing import List, Optional, Tuple

from utils.console import console
from utils.openrouter import OpenRouterClient

from .model import BestFillSelection, FilledDocument, FillRules, WritingStyleSheet
from .prompts import (
    FILL_PROMPT,
    FILL_SYSTEM,
    RULES_PROMPT,
    RULES_SYSTEM,
    SELECT_PROMPT,
    SELECT_SYSTEM,
)
from .utils import format_fill_rules, format_style_sheet, split_sentences

_HUMAN_MATCH_THRESHOLD = 0.85

MAX_RULES_ATTEMPTS = 3
MAX_FILL_ATTEMPTS = 3
N_FILL_CANDIDATES = 3


# ── Step 8: Stochastic span deletion ─────────────────────────────────────────


def create_deletion_mask(
    sentences: List[str],
    stats: dict,
    seed: int,
) -> List[bool]:
    rng = random.Random(seed)

    n = len(sentences)
    if n < 5:
        return [False] * n

    ai_rate = min(
        0.60, max(0.20, rng.gauss(stats["ai_rate_mean"], stats["ai_rate_std"]))
    )
    target_ai = max(1, round(ai_rate * n))

    pool = stats.get("ai_run_lengths") or [1, 1, 2]
    mask = [False] * n
    assigned = 0

    for _ in range(n * 30):
        if assigned >= target_ai:
            break
        span = rng.choice(pool)
        span = min(span, target_ai - assigned)
        if span <= 0:
            continue
        max_start = n - span - 1
        if max_start < 1:
            continue
        start = rng.randint(1, max_start)
        if not any(mask[i] for i in range(start, start + span)):
            for i in range(start, start + span):
                mask[i] = True
            assigned += span

    return mask


# ── Step 9: Per-document fill rules ───────────────────────────────────────────


def generate_fill_rules(
    persona_desc: str,
    doc_lines: List[str],
    client: OpenRouterClient,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RULES_ATTEMPTS + 1):
        try:
            console.log(
                f"  [dim][fill.generate_fill_rules] LLM call attempt {attempt}/{MAX_RULES_ATTEMPTS} → FillRules"
            )
            result: FillRules = client.generate(
                prompt=RULES_PROMPT.format(
                    persona=persona_desc,
                    document="\n".join(doc_lines),
                ),
                output_schema=FillRules,
                system=RULES_SYSTEM,
                temperature=0.7,
            )
            return format_fill_rules(result)
        except Exception as exc:
            last_exc = exc
            console.log(
                f"[yellow]Fill rules attempt {attempt}/{MAX_RULES_ATTEMPTS} failed: "
                f"{type(exc).__name__}: {exc}"
            )

    console.log(
        f"[red]Fill rules generation failed — using persona only. Last error: {last_exc}"
    )
    return ""


# ── Step 11: Best-of-N fill selection ─────────────────────────────────────────


def select_best_fill(
    candidates: List[Tuple[List[str], List[str]]],
    persona: str,
    client: OpenRouterClient,
) -> Tuple[List[str], List[str]]:
    version_blocks = []
    for version_num, (sents, labels) in enumerate(candidates, 1):
        lines = [f"=== VERSION {version_num} ==="]
        for sent, lbl in zip(sents, labels):
            lines.append(f"[{lbl.upper()}] {sent}")
        version_blocks.append("\n".join(lines))

    try:
        console.log(
            f"  [dim][fill.select_best_fill] LLM call {len(candidates)} candidates → BestFillSelection"
        )
        result: BestFillSelection = client.generate(
            prompt=SELECT_PROMPT.format(
                persona=persona,
                n=len(candidates),
                versions="\n\n".join(version_blocks),
            ),
            output_schema=BestFillSelection,
            system=SELECT_SYSTEM,
            temperature=0.0,
        )
        selected = max(1, min(len(candidates), result.best_index)) - 1
        return candidates[selected]
    except Exception as exc:
        console.log(f"[yellow]Selection failed ({exc}), using first candidate")
        return candidates[0]


# ── Steps 10-11: LLM fill ─────────────────────────────────────────────────────


def fill_deleted_sentences(
    sentences: List[str],
    mask: List[bool],
    style_sheet: WritingStyleSheet,
    client: OpenRouterClient,
    persona_desc: str = "",
) -> Tuple[List[str], List[str]]:
    if not any(mask):
        return list(sentences), ["human"] * len(sentences)

    resplit = split_sentences(" ".join(sentences))
    if len(resplit) == len(sentences):
        sentences = resplit

    masked_doc = " ".join(
        "[MASK]" if mask[i] else sentences[i] for i in range(len(sentences))
    )

    effective_persona = persona_desc or format_style_sheet(style_sheet)
    fill_rules = generate_fill_rules(effective_persona, [masked_doc], client)

    prompt = FILL_PROMPT.format(
        persona=effective_persona,
        rules=fill_rules
        if fill_rules
        else "(Use the author persona above as your only guide.)",
        document=masked_doc,
    )

    candidates: List[Tuple[List[str], List[str]]] = []
    last_exc: Optional[Exception] = None

    for cand_num in range(1, N_FILL_CANDIDATES + 1):
        for attempt in range(1, MAX_FILL_ATTEMPTS + 1):
            try:
                console.log(
                    f"  [dim][fill.fill_deleted_sentences] LLM call "
                    f"candidate {cand_num}/{N_FILL_CANDIDATES} attempt {attempt}/{MAX_FILL_ATTEMPTS} "
                    f"→ FilledDocument  (mask={sum(mask)}/{len(sentences)} sentences)"
                )
                result: FilledDocument = client.generate(
                    prompt=prompt,
                    output_schema=FilledDocument,
                    system=FILL_SYSTEM,
                    temperature=0.7,
                )

                output_sents = split_sentences(result.text)
                if not output_sents:
                    raise ValueError("LLM returned an empty document")

                human_originals = [
                    (i, s) for i, s in enumerate(sentences) if not mask[i]
                ]

                def _match_human(s: str, used: set) -> Optional[Tuple[int, str]]:
                    sl = s.strip().lower()
                    for idx, orig in human_originals:
                        if idx in used:
                            continue
                        hl = orig.strip().lower()
                        ratio = difflib.SequenceMatcher(None, sl, hl).ratio()
                        if ratio >= _HUMAN_MATCH_THRESHOLD:
                            return idx, orig
                    return None

                final_sents: List[str] = []
                labels: List[str] = []
                matched_indices: set = set()

                for out_s in output_sents:
                    match = _match_human(out_s, matched_indices)
                    if match is not None:
                        orig_idx, orig = match
                        final_sents.append(orig)
                        labels.append("human")
                        matched_indices.add(orig_idx)
                    else:
                        final_sents.append(out_s)
                        labels.append("AI")

                if len(matched_indices) / len(human_originals) < 0.9:
                    raise ValueError(
                        f"Only {len(matched_indices)}/{len(human_originals)} human sentences "
                        f"found in output"
                    )

                candidates.append((final_sents, labels))
                break

            except Exception as exc:
                last_exc = exc
                console.log(
                    f"[red]Fill attempt {attempt}/{MAX_FILL_ATTEMPTS} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    if not candidates:
        raise RuntimeError(
            f"Fill failed: 0/{N_FILL_CANDIDATES} candidates generated. Last error: {last_exc}"
        )

    if len(candidates) == 1:
        return candidates[0]

    return select_best_fill(candidates, effective_persona, client)
