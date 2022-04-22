from typing import *

import torch
import hydra
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Optimizer


class SiameseHead(nn.Module):

    def __init__(self, in_features: int, final_rep_dim: int):
        super(SiameseHead, self).__init__()
        self.linear = nn.Linear(
            in_features, final_rep_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SiameseModel(pl.LightningModule):

    def __init__(self, input_size: int, **kwargs) -> None:
        super(SiameseModel, self).__init__()
        self.save_hyperparameters()

        self.criterion_label = nn.TripletMarginLoss()
        self.criterion_speaker = nn.TripletMarginLoss()

        self.backbone = hydra.utils.instantiate(self.hparams.backbone, input_size=input_size)

        self.head_label = SiameseHead(self.backbone.embedding_size(), self.hparams.final_dim_rep)
        self.head_speaker = SiameseHead(self.backbone.embedding_size(), self.hparams.final_dim_rep)


    def configure_optimizers(self) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        params = list(self.backbone.parameters()) + list(self.head_label.parameters()) + list(
            self.head_speaker.parameters())

        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=params, _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [optimizer]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        fe = self.backbone(x)
        h1, h2 = self.head_label(fe), self.head_speaker(fe)
        return h1, h2

    def predict(self,
                query: torch.Tensor,
                support: torch.Tensor,
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

    def _inner_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> torch.Tensor:
        label_loss, speaker_loss, total_loss = self._inner_step(batch)

        self.log_dict({
            'train_label_loss': label_loss.item(),
            'train_speaker_loss': speaker_loss.item(),
            'train_total_loss': total_loss.item()
        })
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> torch.Tensor:
        label_loss, speaker_loss, total_loss = self._inner_step(batch)

        self.log_dict({
            'val_label_loss': label_loss.item(),
            'val_speaker_loss': speaker_loss.item(),
            'val_total_loss': total_loss.item()
        })
        return total_loss

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> Dict[str, torch.Tensor]:
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
            f'{role}_accuracy': accuracy_score(y_true, np.where(y_pred < 0.5, 1, 0)),
            f'{role}_f1_score': f1_score(y_true, np.where(y_pred < 0.5, 1, 0)),
            f'{role}_precision_score': precision_score(y_true, np.where(y_pred < 0.5, 1, 0)),
            f'{role}_recall_score': recall_score(y_true, np.where(y_pred < 0.5, 1, 0)),
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
        label_dict = self._calculate_metrics('label', label_trues, np.array(F.sigmoid(torch.tensor(label_preds))))
        speaker_dict = self._calculate_metrics('speaker', speaker_trues, np.array(F.sigmoid(torch.tensor(speaker_preds))))
        self.log_dict(label_dict)
        self.log_dict(speaker_dict)
