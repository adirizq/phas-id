import random

import torch
from torch.utils.data import Dataset


class PHASIDDataset(Dataset):
    def __init__(
        self,
        data,
        meaning_tokenizer,
        K: int = 5,
        num_classes: int = 3,
        token_limit: int = 128,
        support_sentence_limit: int = 128,
        is_train: bool = False,
        preselected_support: dict = {},
    ):
        self.meaning_tokenizer = meaning_tokenizer
        self.K = K
        self.num_classes = num_classes
        self.token_limit = token_limit
        self.support_sentence_limit = support_sentence_limit
        self.is_train = is_train
        self.preselected_support = preselected_support

        # Group docs by author
        author_docs = {}
        for entry in data:
            author = entry["author"]
            if author not in author_docs:
                author_docs[author] = []
            author_docs[author].append(
                {
                    "session_id": entry["session_id"],
                    "sentences": entry["sentence_text"],
                    "labels": entry["label"],
                }
            )

        self.author_docs = author_docs

        # One episode per document (each doc takes a turn as the target)
        self.episodes = []
        for author, docs in author_docs.items():
            for i in range(len(docs)):
                self.episodes.append({"author": author, "target_idx": i})

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        author = ep["author"]
        target_idx = ep["target_idx"]

        all_docs = self.author_docs[author]
        target_doc = all_docs[target_idx]

        # Use pre-selected support (val/test) or random sampling (train)
        if not self.preselected_support:
            sampled_support = self.preselected_support.get(author, {}).get(target_idx, [])
        else:
            available_support = [d for i, d in enumerate(all_docs) if i != target_idx]
            if self.is_train and len(available_support) > self.K:
                sampled_support = random.sample(available_support, self.K)
            else:
                sampled_support = available_support[: self.K]

        # Tokenize target
        target_enc = self.meaning_tokenizer(
            target_doc["sentences"],
            padding="max_length",
            truncation=True,
            max_length=self.token_limit,
            return_tensors="pt",
        )

        support_encoded = []
        for sdoc in sampled_support:
            sentences = sdoc["sentences"][: self.support_sentence_limit]
            labels = sdoc["labels"][: self.support_sentence_limit]
            enc = self.meaning_tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=self.token_limit,
                return_tensors="pt",
            )
            support_encoded.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

        # Pad support docs to the same sentence count (max_mk within episode)
        if support_encoded:
            Kact = len(support_encoded)
            max_mk = max(e["labels"].size(0) for e in support_encoded)

            pad_support_labels = torch.full((Kact, max_mk), fill_value=self.num_classes, dtype=torch.long)
            pad_support_input_ids = torch.zeros(Kact, max_mk, self.token_limit, dtype=torch.long)
            pad_support_attention_masks = torch.zeros(Kact, max_mk, self.token_limit, dtype=torch.long)
            support_lengths = torch.zeros(Kact, dtype=torch.long)

            for i, enc in enumerate(support_encoded):
                mk = enc["labels"].size(0)
                pad_support_labels[i, :mk] = enc["labels"]
                pad_support_input_ids[i, :mk] = enc["input_ids"]
                pad_support_attention_masks[i, :mk] = enc["attention_mask"]
                support_lengths[i] = mk
        else:
            # No support docs available (author has only 1 document)
            pad_support_labels = torch.zeros(0, 0, dtype=torch.long)
            pad_support_input_ids = torch.zeros(0, 0, self.token_limit, dtype=torch.long)
            pad_support_attention_masks = torch.zeros(0, 0, self.token_limit, dtype=torch.long)
            support_lengths = torch.zeros(0, dtype=torch.long)

        n_sents = len(target_doc["sentences"])
        return {
            "target_input_ids": target_enc["input_ids"],
            "target_attention_mask": target_enc["attention_mask"],
            "target_labels": torch.tensor(target_doc["labels"], dtype=torch.long),
            "target_sentences": target_doc["sentences"],
            "support_input_ids": pad_support_input_ids,
            "support_attention_masks": pad_support_attention_masks,
            "support_labels": pad_support_labels,
            "support_lengths": support_lengths,
            "support_sentences": [sdoc["sentences"][: self.support_sentence_limit] for sdoc in sampled_support],
            "support_raw_labels": [sdoc["labels"][: self.support_sentence_limit] for sdoc in sampled_support],
            "session_id": target_doc["session_id"],
            "sent_ids": list(range(n_sents)),
        }


def episode_collate(batch):
    if len(batch) != 1:
        raise ValueError(f"SemiPersonalizedModel requires batch_size=1. Got batch_size={len(batch)}.")
    return batch[0]
