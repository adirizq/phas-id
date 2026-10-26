import json
import os

import lightning as L
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.phas_id.dataset import PHASIDDataset, episode_collate
from utils.console import console, get_progress


class PHASIDDataModule(L.LightningDataModule):
    def __init__(
        self,
        raw_data_path: str,
        parsed_data_path: str,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 1,
        token_limit: int = 128,
        K: int = 5,
        num_classes: int = 3,
        recreate: bool = False,
    ):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.parsed_data_path = parsed_data_path
        self.batch_size = batch_size
        self.token_limit = token_limit
        self.K = K
        self.num_classes = num_classes

        console.rule("[bold yellow]Initializing PHAS-ID DataModule")

        self.meaning_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._doc_emb_cache: dict[str, np.ndarray] = {}
        self._style_model = None

        if not recreate and os.path.exists(self.parsed_data_path):
            console.log(f"[bold green]Found cached metadata at {self.parsed_data_path}")
        else:
            self.parse_data()

    def parse_data(self):
        console.rule(f"[bold]Parsing {self.raw_data_path}")
        os.makedirs(os.path.dirname(self.parsed_data_path), exist_ok=True)

        data = pd.read_csv(self.raw_data_path)
        data["session_id"] = data["session_id"].astype(str)
        data = data[data["label"] != -1]
        data["label"] = data["label"].astype(int)

        split_col = "train_ix" if "train_ix" in data.columns else "original_split"
        grouped = data.groupby("session_id", sort=False)

        console.log(f"Grouping {len(data)} sentences into {grouped.ngroups} sessions...")

        parsed_entries = []
        with get_progress() as progress:
            task = progress.add_task("Grouping sessions...", total=grouped.ngroups)
            for sid, session_data in grouped:
                split = session_data.iloc[0][split_col]
                author = session_data.iloc[0]["worker_id"]
                parsed_entries.append(
                    {
                        "session_id": str(sid),
                        "original_split": int(split) if hasattr(split, "item") else split,
                        "author": int(author) if hasattr(author, "item") else author,
                        "sentence_text": session_data["sentence_text"].astype(str).tolist(),
                        "label": [int(lbl) for lbl in session_data["label"].tolist()],
                    }
                )
                progress.update(task, advance=1)

        console.log("Saving metadata...")
        with open(self.parsed_data_path, "w") as f:
            for entry in parsed_entries:
                f.write(json.dumps(entry) + "\n")

        console.log(f"[bold green]Saved to {self.parsed_data_path}")

    def _preselect_support(self, target_data: list, support_data: list, exclude_self: bool = True) -> dict:
        console.log("Pre-selecting support documents using StyleDistance similarity...")
        if self._style_model is None:
            self._style_model = SentenceTransformer("StyleDistance/styledistance")
        style_model = self._style_model

        support_author_docs: dict[str, list] = {}
        for entry in support_data:
            author = entry["author"]
            if author not in support_author_docs:
                support_author_docs[author] = []
            support_author_docs[author].append(
                {
                    "session_id": entry["session_id"],
                    "sentences": entry["sentence_text"],
                    "labels": entry["label"],
                }
            )

        target_author_docs: dict[str, list] = {}
        for entry in target_data:
            author = entry["author"]
            if author not in target_author_docs:
                target_author_docs[author] = []
            target_author_docs[author].append(
                {
                    "session_id": entry["session_id"],
                    "sentences": entry["sentence_text"],
                }
            )

        all_authors = sorted(set(list(support_author_docs.keys()) + list(target_author_docs.keys())))

        seen_sids: set = set()
        new_docs_to_encode = []
        for author in all_authors:
            for doc in support_author_docs.get(author, []) + target_author_docs.get(author, []):
                if doc["session_id"] not in seen_sids and doc["session_id"] not in self._doc_emb_cache:
                    seen_sids.add(doc["session_id"])
                    new_docs_to_encode.append(doc)

        if new_docs_to_encode:
            all_sentences = []
            doc_slices = []
            for doc in new_docs_to_encode:
                start = len(all_sentences)
                all_sentences.extend(doc["sentences"])
                doc_slices.append((doc["session_id"], start, len(all_sentences)))

            console.log(f"Encoding {len(new_docs_to_encode)} new documents ({len(all_sentences)} sentences) with StyleDistance...")
            all_embs = style_model.encode(
                all_sentences,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=512,
            )

            for sid, s, e in doc_slices:
                self._doc_emb_cache[sid] = all_embs[s:e].mean(axis=0)
        else:
            console.log("All documents already encoded, using cache.")

        preselected: dict[str, dict[int, list]] = {}
        TARGET_BATCH = 512

        for author, target_docs in target_author_docs.items():
            preselected[author] = {}
            support_docs = support_author_docs.get(author, [])

            if not support_docs:
                for target_idx in range(len(target_docs)):
                    preselected[author][target_idx] = []
                continue

            support_embs = np.array([self._doc_emb_cache[doc["session_id"]] for doc in support_docs])
            norms = np.linalg.norm(support_embs, axis=1, keepdims=True) + 1e-8
            normed_support = support_embs / norms  # [N_sup, dim]

            target_embs = np.array([self._doc_emb_cache[doc["session_id"]] for doc in target_docs])
            tnorms = np.linalg.norm(target_embs, axis=1, keepdims=True) + 1e-8
            normed_targets = target_embs / tnorms  # [N_tgt, dim]

            support_sid_to_idx = {doc["session_id"]: i for i, doc in enumerate(support_docs)}

            console.log(f"Selecting top-{self.K} support for {len(target_docs)} targets from {len(support_docs)} candidates...")

            for batch_start in range(0, len(target_docs), TARGET_BATCH):
                batch_end = min(batch_start + TARGET_BATCH, len(target_docs))
                sims_batch = normed_support @ normed_targets[batch_start:batch_end].T

                for local_i, target_idx in enumerate(range(batch_start, batch_end)):
                    target_session_id = target_docs[target_idx]["session_id"]
                    sim_col = sims_batch[:, local_i].copy()

                    if exclude_self:
                        self_idx = support_sid_to_idx.get(target_session_id, -1)
                        if self_idx >= 0:
                            sim_col[self_idx] = -np.inf

                    n_valid = int((sim_col > -np.inf).sum())
                    if n_valid == 0:
                        preselected[author][target_idx] = []
                        continue

                    top_k = min(self.K, n_valid)
                    top_k_pos = np.argpartition(sim_col, -top_k)[-top_k:]
                    top_k_pos = top_k_pos[np.argsort(sim_col[top_k_pos])[::-1]]
                    preselected[author][target_idx] = [support_docs[i] for i in top_k_pos]

        del style_model
        console.log("[bold green]Support pre-selection complete.")
        return preselected

    def setup(self, stage=None):
        self.entries = []
        with open(self.parsed_data_path, "r") as f:
            for line in f:
                self.entries.append(json.loads(line))

        train = [e for e in self.entries if e["original_split"] == "train"]
        val = [e for e in self.entries if e["original_split"] in ["valid", "validation"]]
        test = [e for e in self.entries if e["original_split"] == "test"]

        if stage == "fit" or stage is None:
            train_preselected = self._preselect_support(train, train, exclude_self=True)
            val_preselected = self._preselect_support(val, train + val, exclude_self=True)
            self.train_dataset = PHASIDDataset(
                train,
                self.meaning_tokenizer,
                K=self.K,
                num_classes=self.num_classes,
                token_limit=self.token_limit,
                is_train=True,
                preselected_support=train_preselected,
            )
            self.val_dataset = PHASIDDataset(
                val,
                self.meaning_tokenizer,
                K=self.K,
                num_classes=self.num_classes,
                token_limit=self.token_limit,
                is_train=False,
                preselected_support=val_preselected,
            )
            console.log(f"Train episodes: {len(self.train_dataset)}, Val episodes: {len(self.val_dataset)}")

        if stage == "test":
            test_preselected = self._preselect_support(test, train + val + test, exclude_self=True)
            self.test_dataset = PHASIDDataset(
                test,
                self.meaning_tokenizer,
                K=self.K,
                num_classes=self.num_classes,
                token_limit=self.token_limit,
                is_train=False,
                preselected_support=test_preselected,
            )
            console.log(f"Test episodes: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=episode_collate,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=episode_collate,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=episode_collate,
            num_workers=0,
        )
