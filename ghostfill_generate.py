import argparse
import json
import os
import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.ghostfill import (
    DEFAULT_STATS,
    analyze_coauthor_stats,
    create_deletion_mask,
    document_seed,
    ensure_nltk,
    extract_coauthor_author_stats,
    extract_style_sheets,
    fill_deleted_sentences,
    filter_blogspot,
    match_authors,
    select_authors_and_docs,
)
from utils.console import console
from utils.openrouter import OpenRouterClient

load_dotenv()

def main() -> None:
    parser = argparse.ArgumentParser(description="GhostFill - human-AI collaborative dataset generation")

    parser.add_argument("--k", type=int, default=10, help="Number of Blogspot (X) authors to select")
    parser.add_argument("--docs_per_author", type=int, default=10, help="Documents per author")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, required=True, help="OpenRouter model for LLM steps")
    parser.add_argument("--output_dir", type=str, default="dataset/ghostwriter", help="Base output directory; model name is appended as a subdirectory")
    parser.add_argument("--no_resume", action="store_true", help="Ignore saved intermediate results and recompute everything")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel LLM workers for style sheets and document fill")
    parser.add_argument("--blogspot_path", type=str, default="dataset/blogspot/blogtext.csv", help="Path to the Blogspot CSV (use dataset/blogspot/sample.csv for quick testing)")
    parser.add_argument("--coauthor_path", type=str, default="dataset/coauthor/raw.csv", help="Path to the CoAuthor CSV")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_")
    save_dir = Path(args.output_dir) / model_slug
    save_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_nltk()

    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenRouterClient(api_key=api_key, model=args.model)

    console.log("Loading StyleDistance model…")
    style_model = SentenceTransformer("StyleDistance/styledistance")

    # Step 1
    console.rule("[bold]Step 1 - CoAuthor length statistics")
    coauthor_df = pd.read_csv(args.coauthor_path)
    coauthor_stats = analyze_coauthor_stats(coauthor_df)
    console.log(json.dumps(coauthor_stats, indent=2))
    with open(save_dir / "coauthor_length_stats.json", "w") as f:
        json.dump(coauthor_stats, f, indent=2)

    # Steps 2-3
    console.rule("[bold]Steps 2-3 - Filter Blogspot")
    blogspot_df = pd.read_csv(args.blogspot_path)
    blogspot_filtered = filter_blogspot(blogspot_df, coauthor_stats, min_docs_per_author=10)
    del blogspot_df

    # Step 4
    console.rule(f"[bold]Step 4 - Select {args.k} X authors * {args.docs_per_author} docs")
    author_docs = select_authors_and_docs(blogspot_filtered, args.k, args.docs_per_author, args.seed)
    console.log(f"Selected {len(author_docs)} authors")
    with open(save_dir / "selected_authors.json", "w") as f:
        json.dump(
            {
                "k": args.k,
                "docs_per_author": args.docs_per_author,
                "authors": list(author_docs.keys()),
            },
            f,
            indent=2,
        )

    # Step 5
    console.rule("[bold]Step 5 - Extract Author Writing Style Sheets (delta schema)")
    style_sheets, persona_descriptions = extract_style_sheets(
        author_docs=author_docs,
        client=client,
        save_path=save_dir / "style_sheets.json",
        resume=resume,
        max_workers=args.workers,
    )
    console.log(f"Style sheets ready: {len(style_sheets)}  |  Persona descriptions: {len(persona_descriptions)}")

    # Step 6
    console.rule("[bold]Step 6 - Match X→Y authors via StyleDistance")
    author_matches = match_authors(
        author_docs=author_docs,
        coauthor_df=coauthor_df,
        style_model=style_model,
        save_path=save_dir / "author_matches.json",
        resume=resume,
    )
    console.log(f"Author pairs: {len(author_matches)}")

    # Step 7
    console.rule("[bold]Step 7 - Extract CoAuthor (Y) collaboration statistics")
    coauthor_author_stats = extract_coauthor_author_stats(
        coauthor_df=coauthor_df,
        save_path=save_dir / "coauthor_author_stats.json",
        resume=resume,
    )
    console.log(f"CoAuthor author stats: {len(coauthor_author_stats)} authors")
    del coauthor_df

    # Steps 8-12
    console.rule("[bold]Steps 8-12 - Delete / Fill Rules / Fill / Track / Review")

    output_path = save_dir / "dataset.jsonl"
    done_keys: set = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    done_keys.add((e["blogspot_author_id"], e["doc_index"]))
                except Exception:
                    pass
        console.log(f"  Resuming: {len(done_keys)} entries already written")

    # Build the flat list of tasks up front so we know the total count
    tasks = []
    for blog_id, docs in author_docs.items():
        coauthor_id = author_matches.get(blog_id)
        if not coauthor_id:
            console.log(f"  [yellow]No CoAuthor match for {blog_id}, skipping")
            continue

        stats = coauthor_author_stats.get(str(coauthor_id), DEFAULT_STATS)
        sheet = style_sheets.get(blog_id)
        if sheet is None:
            console.log(f"  [yellow]No style sheet for {blog_id}, skipping")
            continue

        persona = persona_descriptions.get(blog_id, "")

        for doc_idx, sents in enumerate(docs):
            if (blog_id, doc_idx) in done_keys:
                continue
            if len(sents) < 5:
                continue
            tasks.append((blog_id, coauthor_id, doc_idx, sents, stats, sheet, persona))

    console.log(f"  {len(tasks)} documents to process ({len(done_keys)} already done)")

    write_lock = threading.Lock()

    def _fill_one(task):
        blog_id, coauthor_id, doc_idx, sents, stats, sheet, persona = task
        try:
            doc_seed = document_seed(args.seed, blog_id, str(doc_idx))
            mask = create_deletion_mask(sents, stats, seed=doc_seed)
            if not any(mask):
                return None

            try:
                filled_sents, labels = fill_deleted_sentences(sents, mask, sheet, client, persona_desc=persona)
            except RuntimeError as e:
                console.log(f"    [red]Skipping doc (fill failed): {e}")
                return None

            n_h = labels.count("human")
            n_ai = labels.count("AI")
            total = len(labels)

            return {
                "blogspot_author_id": blog_id,
                "coauthor_author_id": coauthor_id,
                "doc_index": doc_idx,
                "sentences": [{"text": sent, "label": lbl} for sent, lbl in zip(filled_sents, labels)],
                "stats": {
                    "total_sentences": total,
                    "n_human": n_h,
                    "n_ai": n_ai,
                    "human_rate": n_h / total if total else 0,
                    "ai_rate": n_ai / total if total else 0,
                },
                "matched_coauthor_stats": {
                    "ai_rate_mean": stats["ai_rate_mean"],
                    "ai_run_length_mean": stats["ai_run_length_mean"],
                    "transition_rate_mean": stats["transition_rate_mean"],
                },
            }

        except Exception:
            console.log(f"  [red]Error: author={blog_id}, doc={doc_idx}\n" + traceback.format_exc())
            return None

    with open(output_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_fill_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Documents"):
                entry = future.result()
                if entry is not None:
                    with write_lock:
                        out_f.write(json.dumps(entry) + "\n")
                        out_f.flush()

    # Summary
    console.rule("[bold green]Dataset complete")
    entries = []
    with open(output_path) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    if entries:
        total_s = sum(e["stats"]["total_sentences"] for e in entries)
        total_h = sum(e["stats"]["n_human"] for e in entries)
        total_ai = sum(e["stats"]["n_ai"] for e in entries)

        console.log(f"Documents  : {len(entries)}")
        console.log(f"Sentences  : {total_s:,}")
        console.log(f"  human    : {total_h:,}  ({100 * total_h / total_s:.1f}%)")
        console.log(f"  AI       : {total_ai:,}  ({100 * total_ai / total_s:.1f}%)")
        console.log(f"Output     : {output_path}")

    client.close()


if __name__ == "__main__":
    main()
