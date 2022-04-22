from typing import *

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.backbones import CustomBackbone


class SiameseHead(nn.Module):

    def __init__(self, config, size):
        super(SiameseHead, self).__init__()
        self.linear = nn.Linear(
            size, 128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class SiameseModel(pl.LightningModule):

    def __init__(self, config, input_size: Tuple[int, int, int]) -> None:
        super(SiameseModel, self).__init__()
        self.config = config
        self.criterion_label = nn.TripletMarginLoss()
        self.criterion_speaker = nn.TripletMarginLoss()
        self.backbone = CustomBackbone(input_size)
        self.head_label = SiameseHead(config, self.backbone.embedding_size())
        self.head_speaker = SiameseHead(config, self.backbone.embedding_size())

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
                support: Optional[torch.Tensor],
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # support is of size [bs, ch, h, w]
        # query is of size [bs, ch, h ,w]
        bs, _, _, _ = support.shape
        bs, _, _, _ = query.shape

        # concat
        samples = torch.cat([query, support]) 
        # pass through backbone
        embedding = self.backbone(samples)

        label_repr = self.head_label(embedding)
        query_label_repr = label_repr[:bs]
        support_label_repr = label_repr[bs:]

        speaker_repr = self.head_label(embedding)
        query_speaker_repr = speaker_repr[:bs]
        support_speaker_repr = speaker_repr[bs:]

        label_distances = F.pairwise_distance(query_label_repr, support_label_repr)
        speaker_distances = F.pairwise_distance(query_speaker_repr, support_speaker_repr)

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

        total_loss = self.config.a * label_loss + self.config.b * speaker_loss

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
        anchor = batch['anchor']
        anchor_label = batch['anchor_label']
        anchor_speaker = batch['anchor_speaker']
        sample = batch['sample']
        sample_label = batch['sample_label']
        sample_speaker = batch['sample_speaker']
        label_target = batch['label_target']
        speaker_target = batch['speaker_target']

        label_distances, speaker_distances = self.predict(
            query=anchor,
            support=sample
        )

        return {
            'anchor_label': anchor_label,
            'anchor_speaker': anchor_speaker,
            'support_label': sample_label,
            'support_speaker': sample_speaker,
            'label_distances': label_distances.cpu(),
            'speaker_distances': speaker_distances.cpu(),
            'label_target': label_target.cpu(),
            'speaker_target': speaker_target.cpu()
        }

    def _calculate_metrics(self, role: str, y_true: Iterable[int], y_pred: np.ndarray) -> Dict[str, float]:
        return {
            f'{role}_accuracy': accuracy_score(y_true, np.where(y_pred < 5, 1, 0)),
            f'{role}_f1_score': f1_score(y_true, np.where(y_pred < 5, 1, 0)),
            f'{role}_precision_score': precision_score(y_true, np.where(y_pred < 5, 1, 0)),
            f'{role}_recall_score': recall_score(y_true, np.where(y_pred < 5, 1, 0)),
        }

    def test_epoch_end(self, outputs: Iterable[Dict[str, torch.Tensor]]) -> None:

        label_trues = []
        speaker_trues = []
        label_preds = []
        speaker_preds = []
        for output in outputs:
            label_trues.extend(output['label_target'])
            speaker_trues.extend(output['speaker_target'])
            label_preds.extend(output['label_distances'])
            speaker_preds.extend(output['speaker_distances'])
        label_dict = self._calculate_metrics('label', label_trues, np.array(label_preds))
        speaker_dict = self._calculate_metrics('speaker', speaker_trues, np.array(speaker_preds))
        self.log_dict(label_dict)
        self.log_dict(speaker_dict)
