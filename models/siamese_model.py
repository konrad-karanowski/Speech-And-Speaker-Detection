from typing import *

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


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
            nn.Conv2d(32, 64, (3, 3))
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
        self.head_label = SiameseHead(config, 9856)
        self.head_speaker = SiameseHead(config, 9856)

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

    def _calculate_pairwise_distances(self, anchor_rep: torch.Tensor, support_rep: Tuple[torch.Tensor]) -> torch.Tensor:
        distances = torch.tensor([
            F.pairwise_distance(anchor_rep, rep) for rep in support_rep
        ])
        return distances

    def _predict(self, query_emb, support, bs):
        support_label_emb = self.backbone(support)
        label_emb = torch.cat([query_emb, support_label_emb])
        label_repr = self.head_label(label_emb).split(bs, dim=0)
        distances = self._calculate_pairwise_distances(
            anchor_rep=label_repr[0],
            support_rep=label_repr[1:]
        )
        predict = torch.argmin(distances)
        proba = 1 / (torch.abs(distances[predict]) + 1e-10)
        return predict, proba

    def predict(self,
                query: torch.Tensor,
                support_label: Optional[torch.Tensor],
                support_speaker: Optional[torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        prediction = {
            'label_pred': None,
            'label_proba': None,
            'speaker_pred': None,
            'speaker_proba': None
        }

        bs = query.shape[0]
        query_emb = self.backbone(query)
        if support_label is not None:
            label_predict, label_proba = self._predict(query_emb, support_label, bs)
            prediction['label_pred'] = label_predict
            prediction['label_proba'] = label_proba

        if support_speaker is not None:
            speaker_predict, speaker_proba = self._predict(query_emb, support_speaker, bs)
            prediction['speaker_pred'] = speaker_predict
            prediction['speaker_proba'] = speaker_proba

        return prediction

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

    def test_step(self, batch, batch_idx, *args, **kwargs) -> None:
        label_loss, speaker_loss, total_loss = self._inner_step(batch)

        self.log_dict({
            'test_label_loss': label_loss.item(),
            'test_speaker_loss': speaker_loss.item(),
            'test_total_loss': total_loss.item()
        })

    def _test_model(self):
        pass
