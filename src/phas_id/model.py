import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SentenceEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", unfreeze_last_n: int = 1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder = self.encoder.float()
        self.hidden_size = self.encoder.config.hidden_size
        self._freeze_layers(unfreeze_last_n)

    def _freeze_layers(self, unfreeze_last_n: int):
        for emb_attr in ["embeddings", "shared", "wte"]:
            module = getattr(self.encoder, emb_attr, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False

        layers = getattr(self.encoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            encoder_module = getattr(self.encoder, "encoder", None)
            layers = getattr(encoder_module, "layer", None) if encoder_module is not None else None

        if layers is None:
            raise AttributeError(f"Could not find transformer layers for {type(self.encoder).__name__}. Expected 'model.layers' (ModernBERT) or 'model.encoder.layer' (BERT/RoBERTa/DeBERTa).")

        total = len(layers)
        for i, layer in enumerate(layers):
            is_unfrozen = i >= total - unfreeze_last_n
            for param in layer.parameters():
                param.requires_grad = is_unfrozen

        for attr in ["final_norm", "pooler", "norm"]:
            module = getattr(self.encoder, attr, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


class IntraDocAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = torch.tensor([])) -> torch.Tensor:
        return self.transformer(x, src_key_padding_mask=padding_mask)


class LabelEncoder(nn.Module):
    def __init__(self, num_classes: int = 3, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(num_classes)
            ]
        )

    def forward(
        self,
        all_embs: torch.Tensor,
        all_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_labels = []
        padding_mask = []

        for c in range(self.num_classes):
            mask = all_labels == c
            if mask.sum() == 0:
                encoded_labels.append(torch.zeros(self.hidden_size, device=all_embs.device, dtype=all_embs.dtype))
                padding_mask.append(True)
            else:
                group_mean = all_embs[mask].mean(dim=0)
                enc_label = self.mlps[c](group_mean)
                encoded_labels.append(enc_label)
                padding_mask.append(False)

        encoded_labels = torch.stack(encoded_labels, dim=0)
        padding_mask = torch.tensor(padding_mask, device=all_embs.device)
        return encoded_labels, padding_mask


class DualCrossAttention(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.macro_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.micro_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        target_embs: torch.Tensor,
        encoded_labels: torch.Tensor,
        kv_mask: torch.Tensor,
        support_embs: torch.Tensor,
        kv_mask_sentences: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        q = target_embs.unsqueeze(0)
        k_macro = encoded_labels.unsqueeze(0)
        v_macro = encoded_labels.unsqueeze(0)

        batch_kv_mask = kv_mask.unsqueeze(0) if kv_mask is not None else None

        macro_attn_out, _ = self.macro_cross_attn(
            q,
            k_macro,
            v_macro,
            key_padding_mask=batch_kv_mask,
        )

        k_micro = support_embs.unsqueeze(0) if support_embs is not None else None
        v_micro = encoded_labels[support_labels].unsqueeze(0) if support_labels is not None else k_micro
        batch_sent_mask = kv_mask_sentences.unsqueeze(0) if kv_mask_sentences is not None else None

        micro_attn_out, _ = self.micro_cross_attn(
            q,
            k_micro,
            v_micro,
            key_padding_mask=batch_sent_mask,
        )

        return self.norm(q + macro_attn_out + micro_attn_out).squeeze(0)


class PHASIDModel(nn.Module):
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_classes: int = 3,
        num_heads: int = 8,
        num_intra_doc_layers: int = 1,
        max_sent_len: int = 128,
        dropout: float = 0.1,
        unfreeze_last_n: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.sentence_encoder = SentenceEncoder(model_name=model_name, unfreeze_last_n=unfreeze_last_n)
        hidden_size = self.sentence_encoder.hidden_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)

        self.intra_doc_attn = IntraDocAttention(hidden_size, num_heads, num_intra_doc_layers, dropout)
        self.label_encoder = LabelEncoder(num_classes, hidden_size, dropout)
        self.dual_cross_attn = DualCrossAttention(hidden_size, num_heads, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _make_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    def _encode_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int = 24,
    ) -> torch.Tensor:
        N = input_ids.size(0)
        if N <= chunk_size:
            return self.sentence_encoder(input_ids, attention_mask)
        embs = []
        for i in range(0, N, chunk_size):
            emb = self.sentence_encoder(input_ids[i : i + chunk_size], attention_mask[i : i + chunk_size])
            embs.append(emb)
        return torch.cat(embs, dim=0)

    def _encode_support(
        self,
        support_input_ids: torch.Tensor,
        support_attention_masks: torch.Tensor,
        support_labels: torch.Tensor,
        support_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        K = support_input_ids.size(0)

        all_embs_list = []
        all_labels_list = []

        text_repr_list = []
        mask_list = []
        labels_padded_list = []

        for k in range(K):
            length = int(support_lengths[k].item())

            with torch.no_grad():
                supp_embds = self._encode_chunked(
                    support_input_ids[k, :length],
                    support_attention_masks[k, :length],
                )
            supp_embds = supp_embds.detach()

            max_mk = support_input_ids.size(1)
            pad_mask = self._make_padding_mask(support_lengths[k : k + 1], max_mk).squeeze(0)  # [max_mk]

            text_embs = torch.zeros(max_mk, self.hidden_size, device=supp_embds.device, dtype=supp_embds.dtype)
            text_embs[:length] = supp_embds

            contextualized = self.intra_doc_attn(
                text_embs.unsqueeze(0),
                padding_mask=pad_mask.unsqueeze(0),
            ).squeeze(0)

            text_repr_list.append(text_embs)
            mask_list.append(pad_mask)

            # Padded label tensor aligned with text_repr_list (padding positions get label 0)
            lbl_padded = torch.zeros(max_mk, dtype=support_labels[k].dtype, device=support_labels.device)
            lbl_padded[:length] = support_labels[k, :length]
            labels_padded_list.append(lbl_padded)

            # Collect only real (non-padded) sentence embeddings and their labels
            all_embs_list.append(contextualized[:length])
            all_labels_list.append(support_labels[k, :length])

        all_embs = torch.cat(all_embs_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        support_text_repr = torch.cat(text_repr_list, dim=0)
        support_mask = torch.cat(mask_list, dim=0)
        support_labels_full = torch.cat(labels_padded_list, dim=0)

        centroids, padding_mask = self.label_encoder(all_embs, all_labels)
        return (
            centroids,
            padding_mask,
            support_text_repr,
            support_mask,
            support_labels_full,
        )

    def forward(
        self,
        batch: dict,
        labels: torch.Tensor = torch.tensor([]),
        class_weights: torch.Tensor = torch.tensor([]),
        return_logits: bool = False,
    ):
        target_input_ids = batch["target_input_ids"]
        target_attention_mask = batch["target_attention_mask"]
        support_input_ids = batch["support_input_ids"]
        support_attention_masks = batch["support_attention_masks"]
        support_labels_ids = batch["support_labels"]
        support_lengths = batch["support_lengths"]

        K = support_labels_ids.size(0) if support_labels_ids is not None else 0
        support_labels_ids = torch.tensor([]) if support_labels_ids is None else support_labels_ids

        target_embs = self.dropout(self._encode_chunked(target_input_ids, target_attention_mask))
        target_embs = self.intra_doc_attn(
            target_embs.unsqueeze(0),
            padding_mask=None,
        ).squeeze(0)

        if K > 0:
            centroids, padding_mask, text_repr_list, mask_list, support_labels_full = self._encode_support(
                support_input_ids,
                support_attention_masks,
                support_labels_ids,
                support_lengths,
            )
            fused = self.dual_cross_attn(
                target_embs,
                centroids,
                padding_mask,
                text_repr_list,
                mask_list,
                support_labels_full,
            )
        else:
            fused = target_embs

        logits = self.mlp(self.dropout(fused)).float()

        if labels.numel() > 0:
            loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=0.05)
            preds = logits.argmax(dim=-1).tolist()
            return loss, preds
        elif return_logits:
            return logits
        else:
            return logits.argmax(dim=-1).tolist()
