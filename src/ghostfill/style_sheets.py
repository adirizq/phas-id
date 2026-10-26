import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.console import console
from utils.openrouter import OpenRouterClient

from .model import BasePost, PersonaProfile, WritingStyleSheet
from .prompts import (
    BASE_PROMPT,
    BASE_SYSTEM,
    COMBINE_PROMPT,
    COMBINE_SYSTEM,
    DELTA_PROMPT,
    DELTA_SYSTEM,
    PERSONA_PROMPT,
    PERSONA_SYSTEM,
)
from .utils import format_style_sheet

_MAX_RETRIES = 3


# Step 5a-i: Base post generation
def generate_base_post(
    doc_sentences: List[str],
    client: OpenRouterClient,
) -> Tuple[str, str]:
    opening = " ".join(doc_sentences[:3])
    wc = len(" ".join(doc_sentences).split())
    console.log(
        f"  [dim][style_sheets.generate_base_post] LLM call → BasePost  (wc={wc})"
    )
    result: BasePost = client.generate(
        prompt=BASE_PROMPT.format(opening=opening, word_count=wc),
        output_schema=BasePost,
        system=BASE_SYSTEM,
        temperature=0.0,
    )
    return result.topic_phrase, result.post


# Step 5a-ii: Delta sheet extraction─


def extract_delta_sheet(
    doc_sentences: List[str],
    base_post: str,
    topic_phrase: str,
    client: OpenRouterClient,
) -> WritingStyleSheet:
    author_text = " ".join(doc_sentences)
    author_words = author_text.split()
    if len(author_words) > 500:
        author_text = " ".join(author_words[:500]) + "..."
    base_words = base_post.split()
    if len(base_words) > 500:
        base_post = " ".join(base_words[:500]) + "..."

    console.log(
        f"  [dim][style_sheets.extract_delta_sheet] LLM call → WritingStyleSheet  (topic='{topic_phrase[:50]}')"
    )
    return client.generate(
        prompt=DELTA_PROMPT.format(
            topic_phrase=topic_phrase,
            base_post=base_post,
            author_post=author_text,
        ),
        output_schema=WritingStyleSheet,
        system=DELTA_SYSTEM,
        temperature=0.0,
    )


# Step 5b: Incremental sheet combination


def combine_sheets(
    prev_sheet: WritingStyleSheet,
    curr_sheet: WritingStyleSheet,
    story_ref: str,
    client: OpenRouterClient,
) -> WritingStyleSheet:
    console.log(
        f"  [dim][style_sheets.combine_sheets] LLM call → WritingStyleSheet  (story_ref={story_ref})"
    )
    return client.generate(
        prompt=COMBINE_PROMPT.format(
            prev_sheet=format_style_sheet(prev_sheet),
            curr_sheet=format_style_sheet(curr_sheet),
            story_ref=story_ref,
        ),
        output_schema=WritingStyleSheet,
        system=COMBINE_SYSTEM,
        temperature=0.0,
    )


# Step 5c: Persona description


def generate_persona(
    sheet: WritingStyleSheet,
    client: OpenRouterClient,
) -> str:
    console.log("  [dim][style_sheets.generate_persona] LLM call → PersonaProfile")
    result: PersonaProfile = client.generate(
        prompt=PERSONA_PROMPT.format(sheet=format_style_sheet(sheet)),
        output_schema=PersonaProfile,
        system=PERSONA_SYSTEM,
        temperature=0.7,
    )
    return result.description


# Persistence


def _save_style_sheets(sheets: Dict[str, WritingStyleSheet], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: v.model_dump() for k, v in sheets.items()}, f, indent=2)


# Per-doc base+delta (parallelisable within an author)


def _generate_delta_for_doc(
    story_idx: int,
    sents: List[str],
    aid: str,
    client: OpenRouterClient,
) -> Tuple[int, Optional[WritingStyleSheet]]:
    """Generate base post + delta sheet for one profiling doc."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            topic_phrase, base_post = generate_base_post(sents, client)
            delta = extract_delta_sheet(sents, base_post, topic_phrase, client)
            return story_idx, delta
        except Exception as exc:
            console.log(
                f"  [yellow]Delta sheet {story_idx} attempt {attempt}/{_MAX_RETRIES} "
                f"failed for author {aid}: {type(exc).__name__}: {exc}"
            )
    console.log(
        f"  [red]Skipping doc {story_idx} for author {aid} after "
        f"{_MAX_RETRIES} failed attempts."
    )
    return story_idx, None


# Per-author pipeline


def _process_one_author(
    aid: str,
    docs: List[List[str]],
    existing_sheet: Optional[WritingStyleSheet],
    client: OpenRouterClient,
) -> Tuple[Optional[WritingStyleSheet], Optional[str]]:
    """
    Full per-author pipeline:
      a. (if no existing sheet) sequential base+delta per doc → sequential combine
      b. persona generation
    Returns (sheet, persona) or (None, None) on unrecoverable failure.
    """
    profiling = docs[: min(5, len(docs))]

    if existing_sheet is not None:
        sheet = existing_sheet
    else:
        # Process profiling docs sequentially — the outer author-level pool
        # already limits total concurrent API calls to max_workers.
        results = [
            _generate_delta_for_doc(i + 1, sents, aid, client)
            for i, sents in enumerate(profiling)
        ]

        delta_sheets = [d for _, d in results if d is not None]

        if not delta_sheets:
            console.log(f"  [red]No delta sheets produced for author {aid}, skipping")
            return None, None

        # Sequential incremental combination (order matters)
        combined = delta_sheets[0]
        for story_idx, delta in enumerate(delta_sheets[1:], 2):
            last_exc: Optional[Exception] = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    combined = combine_sheets(combined, delta, str(story_idx), client)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    console.log(
                        f"  [yellow]Combine step {story_idx} attempt {attempt}/"
                        f"{_MAX_RETRIES} failed for author {aid}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if last_exc is not None:
                console.log(
                    f"  [red]Combine step {story_idx} for author {aid} failed — "
                    f"keeping previous combined sheet."
                )

        sheet = combined

    # Persona generation
    persona: Optional[str] = None
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            persona = generate_persona(sheet, client)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            console.log(
                f"  [yellow]Persona attempt {attempt}/{_MAX_RETRIES} "
                f"failed for author {aid}: {type(exc).__name__}: {exc}"
            )

    if persona is None:
        console.log(
            f"  [red]Persona generation failed for author {aid} — "
            f"falling back to formatted style sheet."
        )
        persona = format_style_sheet(sheet)

    return sheet, persona


# Orchestration


def extract_style_sheets(
    author_docs: Dict[str, List[List[str]]],
    client: OpenRouterClient,
    save_path: Path,
    resume: bool = True,
    max_workers: int = 1,
) -> Tuple[Dict[str, WritingStyleSheet], Dict[str, str]]:
    """
    Delta-schema pipeline with parallelism across authors (max_workers at a time):
      a. Per profiling doc: generate base post → extract delta sheet  (sequential within author)
      b. Combine delta sheets incrementally                            (sequential within author)
      c. Generate persona description

    Returns (sheets, persona_descriptions).
    """
    sheets: Dict[str, WritingStyleSheet] = {}
    personas: Dict[str, str] = {}
    persona_path = save_path.parent / "persona_descriptions.json"
    lock = threading.Lock()

    if resume and save_path.exists():
        with open(save_path) as f:
            raw = json.load(f)
        for aid, data in raw.items():
            try:
                sheets[aid] = WritingStyleSheet.model_validate(data)
            except Exception as exc:
                console.log(f"  [yellow]Could not reload sheet for {aid}: {exc}")
        console.log(f"  Loaded {len(sheets)} existing style sheets")

    if resume and persona_path.exists():
        with open(persona_path) as f:
            personas = json.load(f)
        console.log(f"  Loaded {len(personas)} existing persona descriptions")

    todo = [
        (aid, docs)
        for aid, docs in author_docs.items()
        if aid not in sheets or aid not in personas
    ]

    if not todo:
        return sheets, personas

    def _run_author(
        item: Tuple[str, List[List[str]]],
    ) -> Tuple[str, Optional[WritingStyleSheet], Optional[str]]:
        aid, docs = item
        existing = sheets.get(aid)
        sheet, persona = _process_one_author(aid, docs, existing, client)
        return aid, sheet, persona

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_author, item): item[0] for item in todo}
        with tqdm(total=len(todo), desc="  Style sheets") as pbar:
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    result_aid, sheet, persona = future.result()
                    with lock:
                        if sheet is not None:
                            sheets[result_aid] = sheet
                        if persona is not None:
                            personas[result_aid] = persona
                        # Save both atomically under the same lock acquisition
                        if sheet is not None:
                            _save_style_sheets(sheets, save_path)
                        if persona is not None:
                            persona_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(persona_path, "w") as f:
                                json.dump(personas, f, indent=2)
                except Exception as exc:
                    console.log(f"  [red]Unexpected error for author {aid}: {exc}")
                finally:
                    pbar.update(1)

    return sheets, personas
