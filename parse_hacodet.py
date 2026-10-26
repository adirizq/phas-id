import argparse
import json
import os
import re
from collections import Counter
from typing import cast
from uuid import uuid4

import nltk
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

from utils.console import ConsoleColors, console, get_progress


class HACoDetPreprocessor:
    def __init__(
        self,
        raw_data_path: str,
        parsed_data_path: str,
    ):
        self.raw_data_path = raw_data_path
        self.parsed_data_path = parsed_data_path

    def __load_nltk(self):
        try:
            nltk.data.find("tokenizers/punkt/english.pickle")
        except LookupError:
            console.log("Downloading punkt...")
            nltk.download("punkt")

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            console.log("Downloading punkt_tab...")
            nltk.download("punkt_tab")

        self._detokenizer = TreebankWordDetokenizer()

    def __get_most_common_tag(self, tags):

        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0][0]
        return most_common_tag

    def __split_sentence(self, sentence):

        pattern = re.compile(r"\S+|\s")
        words = pattern.findall(sentence)
        return words

    def parse(self, chunk_size: int = 128) -> None:
        console.rule(f"[bold]Parsing {self.raw_data_path}")

        console.log("[bold]Start parsing...")
        os.makedirs(os.path.dirname(self.parsed_data_path), exist_ok=True)
        self.__load_nltk()

        sent_separator = cast(nltk.tokenize.PunktSentenceTokenizer, nltk.data.load("tokenizers/punkt/english.pickle"))

        models = os.listdir(self.raw_data_path)

        valid_models = []
        for model in models:
            model_path = os.path.join(self.raw_data_path, model)
            if os.path.isdir(model_path) and len(os.listdir(model_path)) > 0:
                if any(file_name.endswith(".json") for file_name in os.listdir(model_path)):
                    valid_models.append(model)

        models = valid_models

        files = []
        for model in models:
            for file_name in os.listdir(os.path.join(self.raw_data_path, model)):
                files.append(os.path.join(self.raw_data_path, model, file_name))

        raw_docs = []

        for file_prog, file in enumerate(files, 1):
            try:
                file_name = file.split("/")[-1]
                domain = file_name.split("_")[0]
                model = file_name.split("_")[1]
                original_split = file_name.split("_")[2].split(".")[0]

                data_temp = json.load(open(file))
                console.log(f"[bold][{file_prog}/{len(files)}] Parsing {file_name}")

                with get_progress() as progress:
                    task_id = progress.add_task("Progress", total=len(data_temp))
                    for data in data_temp:
                        words = data["text"]
                        labels = data["label"]
                        text = " ".join(words)
                        sentences = sent_separator.tokenize(text)

                        doc_sentences = []
                        doc_labels = []
                        offset = 0
                        start_word_idx = 0

                        for sent in sentences:
                            start = text[offset:].find(sent) + offset
                            end = start + len(sent)
                            offset = end

                            sent_words = [w for w in self.__split_sentence(text[start:end]) if w != " "]
                            end_word_idx = start_word_idx + len(sent_words)
                            tags = labels[start_word_idx:end_word_idx]
                            if not tags:
                                continue
                            sent_token_slice = words[start_word_idx:end_word_idx]
                            start_word_idx = end_word_idx

                            doc_sentences.append(self._detokenizer.detokenize(sent_token_slice))
                            doc_labels.append(self.__get_most_common_tag(tags))

                        if doc_sentences:
                            raw_docs.append(
                                {
                                    "sentences": doc_sentences,
                                    "labels": doc_labels,
                                    "domain": domain,
                                    "model": model,
                                    "split": original_split,
                                }
                            )
                        progress.update(task_id, advance=1)

            except Exception as e:
                console.log(f"[bold red]Error parsing {file}: {e}")
                continue

        console.log(f"[bold green]Using chunk_size={chunk_size}")

        parsed_data = {
            "id": [],
            "session_id": [],
            "label": [],
            "sentence_text": [],
            "domain": [],
            "model": [],
            "original_split": [],
            "word_count": [],
            "worker_id": [],
        }

        for doc in raw_docs:
            doc_uuid = uuid4()
            sents = doc["sentences"]
            labels = doc["labels"]
            for chunk_idx, chunk_start in enumerate(range(0, len(sents), chunk_size)):
                chunk_sents = sents[chunk_start : chunk_start + chunk_size]
                chunk_labels = labels[chunk_start : chunk_start + chunk_size]
                chunk_session_id = f"{doc_uuid}_{chunk_idx}"
                for sent_idx, (sent, label) in enumerate(zip(chunk_sents, chunk_labels)):
                    parsed_data["id"].append(f"{doc_uuid}_{chunk_idx}_{sent_idx}")
                    parsed_data["session_id"].append(chunk_session_id)
                    parsed_data["label"].append(label)
                    parsed_data["sentence_text"].append(sent)
                    parsed_data["domain"].append(doc["domain"])
                    parsed_data["model"].append(doc["model"])
                    parsed_data["original_split"].append(doc["split"])
                    parsed_data["word_count"].append(len(sent.split()))
                    parsed_data["worker_id"].append("global")

        parsed_data_df = pd.DataFrame(parsed_data)
        parsed_data_df.to_csv(self.parsed_data_path, index=False)
        console.log(f"[bold {ConsoleColors.green}]Parsed {len(parsed_data_df)} sentences ({len(raw_docs)} docs → chunk_size={chunk_size}) to {self.parsed_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", "-ra_p", type=str, default="dataset/hacodet/raw", help="Path to the raw data")
    parser.add_argument("--parsed_data_path", "-pa_p", type=str, default="dataset/hacodet/raw.csv", help="Path to the parsed data")
    parser.add_argument("--chunk_size", "-cs", type=int, required=True, help="Sentences per chunk")
    args = parser.parse_args()

    preprocessor = HACoDetPreprocessor(
        raw_data_path=args.raw_data_path,
        parsed_data_path=args.parsed_data_path,
    )

    preprocessor.parse(chunk_size=args.chunk_size)
