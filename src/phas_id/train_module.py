import json
import os

import lightning as L
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import get_cosine_schedule_with_warmup

from src.phas_id.model import PHASIDModel


class PHASIDTrainModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_classes: int = 3,
        K: int = 5,
        num_heads: int = 8,
        num_intra_doc_layers: int = 3,
        max_sent_len: int = 128,
        unfreeze_last_n: int = 2,
        dropout: float = 0.1,
        lr: float = 2e-5,
        run_name: str = "default",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.run_name = run_name
        self.lr = lr

        self.classifier = PHASIDModel(
            model_name=model_name,
            num_classes=num_classes,
            num_heads=num_heads,
            num_intra_doc_layers=num_intra_doc_layers,
            max_sent_len=max_sent_len,
            unfreeze_last_n=unfreeze_last_n,
            dropout=dropout,
        )

        self.train_step_outputs = {"targets": [], "predictions": []}
        self.validation_step_outputs = {"targets": [], "predictions": []}
        self.test_step_outputs = {"targets": [], "predictions": [], "ids": []}

        self.class_labels = ["Human", "AI", "Human-AI"]

        if num_classes == 3:
            weights = torch.tensor([1.0, 2.5, 2.5])
        else:
            weights = torch.tensor([1.0, 2.5])
        self.register_buffer("class_weights", weights)

    def forward(self, batch, labels=None, return_logits=False):
        weights = self.class_weights if labels is not None else None
        result = self.classifier(batch, labels, class_weights=weights, return_logits=return_logits)
        if labels is not None:
            loss, preds = result
            return loss, preds
        return result

    def configure_optimizers(self):
        encoder_param_ids = {id(p) for p in self.classifier.sentence_encoder.parameters()}
        encoder_params = [p for p in self.classifier.sentence_encoder.parameters()]
        new_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(50, int(0.1 * total_steps))

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.lr, "weight_decay": 0.01},
                {"params": new_params, "lr": self.lr * 5, "weight_decay": 0.1},
            ]
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        labels = batch["target_labels"]
        loss, prediction_paths = self(batch, labels=labels)

        self.train_step_outputs["targets"].extend(labels.cpu().numpy().tolist())
        self.train_step_outputs["predictions"].extend(prediction_paths)

        self.log(
            "train/loss",
            loss,
            batch_size=1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_train_epoch_end(self):
        all_targets = self.train_step_outputs["targets"]
        all_predictions = self.train_step_outputs["predictions"]
        if len(all_targets) == 0:
            return
        metrics = self.calculate_metrics(all_targets, all_predictions)
        log_dict = {
            "train/accuracy": metrics["accuracy"],
            "train/macro_f1_score": metrics["macro_f1"],
            "train/weighted_f1_score": metrics["weighted_f1"],
        }
        for lbl, scores in metrics["per_class"].items():
            name = self.class_labels[lbl] if lbl < len(self.class_labels) else str(lbl)
            log_dict[f"train/{name}_f1"] = scores["f1"]
            log_dict[f"train/{name}_precision"] = scores["precision"]
            log_dict[f"train/{name}_recall"] = scores["recall"]
        self.log_dict(log_dict, prog_bar=True, logger=True)
        self.train_step_outputs["targets"].clear()
        self.train_step_outputs["predictions"].clear()

    def validation_step(self, batch, batch_idx):
        labels = batch["target_labels"]
        loss, prediction_paths = self(batch, labels=labels)

        self.validation_step_outputs["targets"].extend(labels.cpu().numpy().tolist())
        self.validation_step_outputs["predictions"].extend(prediction_paths)

        self.log(
            "val/loss",
            loss,
            batch_size=1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_validation_epoch_end(self):
        all_targets = self.validation_step_outputs["targets"]
        all_predictions = self.validation_step_outputs["predictions"]
        if len(all_targets) == 0 or self.trainer.sanity_checking:
            return
        metrics = self.calculate_metrics(all_targets, all_predictions)
        log_dict = {
            "val/accuracy": metrics["accuracy"],
            "val/macro_f1_score": metrics["macro_f1"],
            "val/weighted_f1_score": metrics["weighted_f1"],
        }
        for lbl, scores in metrics["per_class"].items():
            name = self.class_labels[lbl] if lbl < len(self.class_labels) else str(lbl)
            log_dict[f"val/{name}_f1"] = scores["f1"]
            log_dict[f"val/{name}_precision"] = scores["precision"]
            log_dict[f"val/{name}_recall"] = scores["recall"]
        self.log_dict(log_dict, prog_bar=True, logger=True)
        self.validation_step_outputs["targets"].clear()
        self.validation_step_outputs["predictions"].clear()

    def test_step(self, batch, batch_idx):
        labels = batch["target_labels"]

        if getattr(self, "binary_output", False):
            logits = self.classifier(batch, return_logits=True)
            probs = torch.softmax(logits.float(), dim=-1)
            prediction_paths = (probs[:, 1] > probs[:, 0]).long().cpu().numpy().tolist()
        else:
            prediction_paths = self.classifier(batch)

        session_id = batch["session_id"]
        sent_ids = batch["sent_ids"]
        self.test_step_outputs["ids"].extend([f"{session_id}_{i}" for i in sent_ids])
        self.test_step_outputs["targets"].extend(labels.cpu().numpy().tolist())
        self.test_step_outputs["predictions"].extend(prediction_paths)

    def on_test_epoch_end(self):
        all_targets = self.test_step_outputs["targets"]
        all_predictions = self.test_step_outputs["predictions"]
        if len(all_targets) == 0:
            return

        binary_output = getattr(self, "binary_output", False)
        metrics = self.calculate_metrics(
            all_targets,
            all_predictions,
            num_classes=2 if binary_output else self.num_classes,
        )
        log_dict = {
            "test/accuracy": metrics["accuracy"],
            "test/macro_f1_score": metrics["macro_f1"],
            "test/weighted_f1_score": metrics["weighted_f1"],
        }
        for lbl, scores in metrics["per_class"].items():
            name = self.class_labels[lbl] if lbl < len(self.class_labels) else str(lbl)
            log_dict[f"test/{name}_f1"] = scores["f1"]
            log_dict[f"test/{name}_precision"] = scores["precision"]
            log_dict[f"test/{name}_recall"] = scores["recall"]
        self.log_dict(log_dict, prog_bar=True, logger=True)

        if len(all_targets) > 0:
            output_dir = f"logs/phas_id/{self.run_name}"
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(
                {
                    "id": self.test_step_outputs["ids"],
                    "target": all_targets,
                    "prediction": all_predictions,
                }
            )
            df.to_csv(f"{output_dir}/test_predictions.csv", index=False)
            with open(f"{output_dir}/test_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

        self.test_step_outputs["ids"].clear()
        self.test_step_outputs["targets"].clear()
        self.test_step_outputs["predictions"].clear()

    def calculate_metrics(self, targets, predictions, num_classes: int = 3):
        report: dict = classification_report(targets, predictions, output_dict=True, zero_division=0)  # type: ignore
        per_class = {}
        for i in range(num_classes if num_classes is not None else self.num_classes):
            str_i = str(i)
            if str_i in report:
                per_class[i] = {
                    "f1": report[str_i]["f1-score"] * 100,
                    "precision": report[str_i]["precision"] * 100,
                    "recall": report[str_i]["recall"] * 100,
                }
            else:
                per_class[i] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        return {
            "accuracy": report["accuracy"] * 100,
            "macro_f1": report["macro avg"]["f1-score"] * 100,
            "weighted_f1": report["weighted avg"]["f1-score"] * 100,
            "per_class": per_class,
        }
