import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.console import console

from .utils import split_sentences, word_count


# Step 1: CoAuthor length statistics
def analyze_coauthor_stats(df) -> dict:
    sent_per_doc = df.groupby("session_id")["sent_id"].max()
    word_per_doc = df.groupby("session_id")["word_count"].sum()
    return {
        "sentence_count": {
            "min": int(sent_per_doc.min()),
            "max": int(sent_per_doc.max()),
            "mean": float(sent_per_doc.mean()),
            "median": float(sent_per_doc.median()),
            "p25": float(sent_per_doc.quantile(0.25)),
            "p75": float(sent_per_doc.quantile(0.75)),
        },
        "word_count": {
            "min": int(word_per_doc.min()),
            "max": int(word_per_doc.max()),
            "mean": float(word_per_doc.mean()),
            "median": float(word_per_doc.median()),
            "p25": float(word_per_doc.quantile(0.25)),
            "p75": float(word_per_doc.quantile(0.75)),
        },
    }


# Steps 2-3: Filter Blogspot
def filter_blogspot(
    df,
    coauthor_stats: dict,
    min_docs_per_author: int = 10,
) -> pd.DataFrame:
    df = df.copy()
    df["word_count"] = df["text"].apply(word_count)

    wc_min = int(coauthor_stats["word_count"]["p25"])
    wc_max = int(coauthor_stats["word_count"]["max"])

    console.log(f"  Word-count filter: {wc_min}–{wc_max}")
    df = df[(df["word_count"] >= wc_min) & (df["word_count"] <= wc_max)]

    # First char is ASCII likely English
    df = df[df["text"].apply(lambda t: len(t) > 0 and t[0].isascii())]
    console.log(f"  After length+language filter: {len(df):,} docs")

    counts = df.groupby("id").size()
    valid = counts[counts >= min_docs_per_author].index
    df = df[df["id"].isin(valid)]
    console.log(
        f"  After ≥{min_docs_per_author} docs/author filter: {len(df):,} docs, {len(valid):,} authors"
    )

    return df.reset_index(drop=True)


# Step 4: Select K authors and docs
def select_authors_and_docs(
    df,
    k: int,
    docs_per_author: int,
    seed: int,
    min_sentences: int = 21,
    max_sentences: int = 78,
) -> Dict[str, List[List[str]]]:
    rng = random.Random(seed)
    authors = df["id"].unique().tolist()
    rng.shuffle(authors)

    author_docs: Dict[str, List[List[str]]] = {}
    for aid in authors:
        if len(author_docs) == k:
            break
        texts = df[df["id"] == aid]["text"].tolist()
        all_sents = [split_sentences(t) for t in texts]
        qualified = [s for s in all_sents if min_sentences <= len(s) <= max_sentences]
        if len(qualified) < docs_per_author:
            continue
        sampled = rng.sample(qualified, docs_per_author)
        author_docs[str(aid)] = sampled

    return author_docs


# Step 6: Match X authors to Y (CoAuthor) authors via StyleDistance
def match_authors(
    author_docs: Dict[str, List[List[str]]],
    coauthor_df,
    style_model: SentenceTransformer,
    save_path: Path,
    resume: bool = True,
    batch_size: int = 64,
) -> Dict[str, str]:
    if resume and save_path.exists():
        with open(save_path) as f:
            matches = json.load(f)
        console.log(f"  Loaded {len(matches)} existing author matches")
        return matches

    console.log("  Embedding Blogspot authors…")
    blog_vecs: Dict[str, np.ndarray] = {}
    for aid, docs in tqdm(author_docs.items(), desc="  Embedding X authors"):
        sents = [s for doc in docs for s in doc if len(s) > 10][:100]
        if not sents:
            continue
        emb = style_model.encode(
            sents, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True
        )
        blog_vecs[aid] = emb.mean(axis=0)

    console.log("  Embedding CoAuthor authors (human sentences only)…")
    human_df = coauthor_df[coauthor_df["label"] == 0].copy()
    coauthor_vecs: Dict[str, np.ndarray] = {}
    for aid, grp in tqdm(human_df.groupby("worker_id"), desc="  Embedding Y authors"):
        sents = grp["sentence_text"].tolist()[:100]
        if not sents:
            continue
        emb = style_model.encode(
            sents, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True
        )
        coauthor_vecs[str(aid)] = emb.mean(axis=0)

    coauthor_ids = list(coauthor_vecs.keys())
    C = np.array([coauthor_vecs[a] for a in coauthor_ids])
    C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)

    matches: Dict[str, str] = {}
    for bid, bvec in blog_vecs.items():
        bvec_norm = bvec / (np.linalg.norm(bvec) + 1e-8)
        sims = C_norm @ bvec_norm
        matches[bid] = coauthor_ids[int(np.argmax(sims))]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(matches, f, indent=2)
    console.log(f"  Saved author matches → {save_path}")
    return matches


# Step 7: CoAuthor per-author stats

DEFAULT_STATS = {
    "ai_rate_mean": 0.20,
    "ai_rate_std": 0.10,
    "ai_run_length_mean": 1.5,
    "ai_run_length_std": 1.0,
    "ai_run_lengths": [1, 1, 1, 2, 2, 3],
    "transition_rate_mean": 0.20,
    "transition_rate_std": 0.05,
}


def extract_coauthor_author_stats(
    coauthor_df,
    save_path: Path,
    resume: bool = True,
) -> Dict[str, dict]:
    if resume and save_path.exists():
        with open(save_path) as f:
            stats = json.load(f)
        console.log(f"  Loaded CoAuthor stats for {len(stats)} authors")
        return stats

    data = coauthor_df[coauthor_df["label"].isin([0, 1])].copy()
    author_stats: Dict[str, dict] = {}

    for author_id, adf in tqdm(data.groupby("worker_id"), desc="  CoAuthor stats"):
        session_results = []

        for _, sdf in adf.groupby("session_id"):
            labels = sdf.sort_values("sent_id")["label"].tolist()
            if len(labels) < 3:
                continue

            ai_runs: List[int] = []
            human_runs: List[int] = []
            cur, cnt = labels[0], 1
            for lbl in labels[1:]:
                if lbl == cur:
                    cnt += 1
                else:
                    (ai_runs if cur == 1 else human_runs).append(cnt)
                    cur, cnt = lbl, 1
            (ai_runs if cur == 1 else human_runs).append(cnt)

            transitions = sum(
                1 for i in range(len(labels) - 1) if labels[i] != labels[i + 1]
            )
            ai_count = sum(1 for lbl in labels if lbl == 1)

            session_results.append(
                dict(
                    n=len(labels),
                    transitions=transitions,
                    ai_count=ai_count,
                    ai_runs=ai_runs,
                    human_runs=human_runs,
                )
            )

        if not session_results:
            continue

        all_ai_runs = [r for s in session_results for r in s["ai_runs"]]
        all_human_runs = [r for s in session_results for r in s["human_runs"]]
        tr_rates = [s["transitions"] / s["n"] for s in session_results]
        ai_rates = [s["ai_count"] / s["n"] for s in session_results]

        author_stats[str(author_id)] = {
            "transition_rate_mean": float(np.mean(tr_rates)),
            "transition_rate_std": float(np.std(tr_rates)),
            "ai_rate_mean": float(np.mean(ai_rates)),
            "ai_rate_std": float(np.std(ai_rates)),
            "ai_run_length_mean": float(np.mean(all_ai_runs)) if all_ai_runs else 1.0,
            "ai_run_length_std": float(np.std(all_ai_runs)) if all_ai_runs else 0.0,
            "ai_run_lengths": [int(r) for r in all_ai_runs] or [1],
            "human_run_length_mean": float(np.mean(all_human_runs))
            if all_human_runs
            else 1.0,
            "human_run_lengths": [int(r) for r in all_human_runs] or [1],
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(author_stats, f, indent=2)
    console.log(f"  Saved stats for {len(author_stats)} CoAuthor authors → {save_path}")
    return author_stats
