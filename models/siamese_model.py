from typing import *

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SiameseHead(nn.Module):

    def __init__(self, config, size):
        super(SiameseHead, self).__init__()
        self.linear = nn.Linear(
            size, 128
        )

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)


class Backbone(nn.Module):

    def __init__(self, config) -> None:
        super(Backbone, self).__init__()
        x, y = config.input_shape
        self.fe = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.MaxPool2d(3),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fe = self.fe(x)
        flatt = self.flatten(fe)
        return flatt


class SiameseModel(pl.LightningModule):

    def __init__(self, config) -> None:
        super(SiameseModel, self).__init__()
        self.config = config
        self.criterion_label = nn.TripletMarginLoss()
        self.criterion_speaker = nn.TripletMarginLoss()
        self.backbone = Backbone(config)
        self.head_label = SiameseHead(config, 768)
        self.head_speaker = SiameseHead(config, 768)

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.head_label.parameters()) + list(
            self.head_speaker.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        fe = self.backbone(x)
        h1, h2 = self.head_label(fe), self.head_speaker(fe)
        return h1, h2

    def _calculate_distances(self, query_repr: torch.Tensor, support_repr: torch.Tensor, p: int = 2) -> torch.Tensor:
        return torch.cdist(query_repr.unsqueeze(0), support_repr.unsqueeze(0), p=p).squeeze(0)

    def predict(self,
                query: torch.Tensor,
                support_label: Optional[torch.Tensor],
                support_speaker: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # support is of size [num_classes, ch, h, w]
        num_labels, ch, h, w = support_label.shape
        num_speakers, ch, h, w = support_speaker.shape
        bs, ch, h, w = query.shape
        # concat everything
        samples = torch.cat([query, support_label, support_speaker], dim=0)
        # pass through backbone
        embedding = self.backbone(samples)

        query_emb, label_emb, speaker_emb = \
            embedding[:bs], embedding[bs:(bs + num_labels)], embedding[(bs + num_labels):]

        label_repr = self.head_label(
            torch.cat([query_emb, label_emb], dim=0)
        )
        query_label_repr = label_repr[:bs]
        support_label_repr = label_repr[bs:]

        speaker_repr = self.head_label(
            torch.cat([query_emb, speaker_emb], dim=0)
        )
        query_speaker_repr = speaker_repr[:bs]
        support_speaker_repr = speaker_repr[bs:]

        label_distances = self._calculate_distances(query_label_repr, support_label_repr, p=2)
        speaker_distances = self._calculate_distances(query_speaker_repr, support_speaker_repr, p=2)

        return label_distances, speaker_distances

    def _inner_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, pos_label, neg_label, pos_speaker, neg_speaker = batch['anchor'], batch['pos_label'], \
                                                                 batch['neg_label'], batch['pos_speaker'], batch[
                                                                     'neg_speaker']
        bs = anchor.shape[0]
        x = torch.cat([pos_label, neg_label, anchor, pos_speaker, neg_speaker], dim=0)
        embeddings = self.backbone(x)
        label_emb, speaker_emb = embeddings[:3 * bs, :], embeddings[2 * bs:, :]
        label_repr = self.head_label(label_emb)
        speaker_repr = self.head_speaker(speaker_emb)

        pos_lab_rep, neg_lab_rep, anchor_lab_rep = label_repr.split(bs, dim=0)
        anchor_sp_rep, pos_sp_rep, neg_sp_rep = speaker_repr.split(bs, dim=0)

        label_loss = self.criterion_label(anchor_lab_rep, pos_lab_rep, neg_lab_rep)
        speaker_loss = self.criterion_speaker(anchor_sp_rep, pos_sp_rep, neg_sp_rep)

        total_loss = label_loss + speaker_loss

        return label_loss, speaker_loss, total_loss

    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        label_loss, speaker_loss, total_loss = self._inner_step(batch)

        self.log_dict({
            'train_label_loss': label_loss.item(),
            'train_speaker_loss': speaker_loss.item(),
            'train_total_loss': total_loss.item()
        })
        return total_loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        label_loss, speaker_loss, total_loss = self._inner_step(batch)

        self.log_dict({
            'val_label_loss': label_loss.item(),
            'val_speaker_loss': speaker_loss.item(),
            'val_total_loss': total_loss.item()
        })
        return total_loss

    def test_step(self, batch, batch_idx, *args, **kwargs) -> Dict[str, torch.Tensor]:
        anchor, pos_label, neg_label, pos_speaker, neg_speaker = batch['anchor'], batch['pos_label'], \
                                                                 batch['neg_label'], batch['pos_speaker'], batch[
                                                                     'neg_speaker']
        batch_size = anchor.shape[0]
        support_label = torch.cat([pos_label[0, :], neg_label[0, :]], dim=0)[:, None]
        support_speaker = torch.cat([pos_speaker[0, :], neg_speaker[0, :]], dim=0)[:, None]
        label_distances, speaker_distances = self.predict(
            query=anchor,
            support_label=support_label,
            support_speaker=support_speaker
        )

        true_labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).cpu()
        binary_label_preds = label_distances.transpose(0, 1).reshape(-1).cpu()
        binary_speaker_preds = speaker_distances.transpose(0, 1).reshape(-1).cpu()
        return {
            'true': true_labels,
            'label_preds': binary_label_preds,
            'speaker_preds': binary_speaker_preds
        }

    def _calculate_metrics(self, role: str, y_true: Iterable[int], y_pred: np.ndarray) -> Dict[str, float]:
        return {
            f'{role}_accuracy': accuracy_score(y_true, np.where(y_pred < 5, 1, 0)),
            f'{role}_f1_score': f1_score(y_true, np.where(y_pred < 5, 1, 0)),
            f'{role}_precision_score': precision_score(y_true, np.where(y_pred < 5, 1, 0)),
            f'{role}_recall_score': recall_score(y_true, np.where(y_pred < 5, 1, 0)),
        }

    def test_epoch_end(self, outputs: Iterable[Dict[str, torch.Tensor]]) -> None:

        trues = []
        label_preds = []
        speaker_preds = []
        for output in outputs:
            trues.extend(output['true'])
            label_preds.extend(output['label_preds'])
            speaker_preds.extend(output['speaker_preds'])
        label_dict = self._calculate_metrics('label', trues, np.array(label_preds))
        speaker_dict = self._calculate_metrics('speaker', trues, np.array(speaker_preds))
        self.log_dict(label_dict)
        self.log_dict(speaker_dict)
