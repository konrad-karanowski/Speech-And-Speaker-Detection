from typing import *

import hydra
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from torch.optim import Optimizer


class SiameseHead(nn.Module):

    def __init__(self, in_features: int, final_rep_dim: int):
        """Base "head" for siamese model. Transforms backbone's output into specific vector. Responsible for one dimension (speaker, label).

        Args:
            in_features (int): Embedding size form backbone.
            final_rep_dim (int): Final vector size. 
        """
        super(SiameseHead, self).__init__()
        self.linear = nn.Linear(
            in_features, final_rep_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SiameseModel(pl.LightningModule):

    def __init__(self, input_size: int, **kwargs) -> None:
        """Multi-Head Siamese model for pre-training. Learns acoustic embeddings of word and speaker.
        This model uses TripletMarginLoss as criterion.

        Args:
            input_size (int): Backbone input size.
        """
        super(SiameseModel, self).__init__()
        self.save_hyperparameters()

        self.criterion_label = nn.TripletMarginLoss()
        self.criterion_speaker = nn.TripletMarginLoss()

        self.backbone = hydra.utils.instantiate(self.hparams.backbone, input_size=input_size)

        self.head_label = SiameseHead(self.backbone.embedding_size(), self.hparams.final_dim_rep)
        self.head_speaker = SiameseHead(self.backbone.embedding_size(), self.hparams.final_dim_rep)

        # for model inference
        self.preprocess_method = hydra.utils.instantiate(self.hparams.process_audio_method, _partial_=True)
        self.spectogram_method = hydra.utils.instantiate(self.hparams.spectrogram_method, _partial_=True)

    def configure_optimizers(self) -> Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Configure optimizer and lr scheduler.

        Returns:
            Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]: Optimizer or optimizer and lr scheduler.
        """
        params = list(self.backbone.parameters()) + list(self.head_label.parameters()) + list(
            self.head_speaker.parameters())

        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=params, _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [optimizer]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=optimizer)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.hparams.monitor}
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Base torch interface for model forward-propagation.

        Args:
            x (torch.Tensor): Input for the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Label embedding and speaker embedding.
        """
        fe = self.backbone(x)
        label_embedding, speaker_embedding = self.head_label(fe), self.head_speaker(fe)
        return label_embedding, speaker_embedding

    def _inner_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inner step of calculating embeddings of speaker and label.

        Args:
            batch (Any): Input data including anchor, positive and negative samples across all dimensions (label, speaker).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Label triplet margin loss, speaker triplet margin loss and total loss (sum).
        """
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

    def training_step(self, batch: Any, *args, **kwargs) -> torch.Tensor:
        """Base PyTorchLightning step for training process.

        Args:
            batch (Any): Sample of training data.

        Returns:
            torch.Tensor: Training loss.
        """
        label_loss, speaker_loss, total_loss = self._inner_step(batch)

        self.log_dict({
            'train_label_loss': label_loss.item(),
            'train_speaker_loss': speaker_loss.item(),
            'train_total_loss': total_loss.item()
        })
        return total_loss

    def validation_step(self, batch: Any, *args, **kwargs) -> torch.Tensor:
        """Base PyTorchLightning step for validation process.

        Args:
            batch (Any): Sample of validation data.

        Returns:
            torch.Tensor: Validation loss.
        """
        label_loss, speaker_loss, total_loss = self._inner_step(batch)

        self.log_dict({
            'val_label_loss': label_loss.item(),
            'val_speaker_loss': speaker_loss.item(),
            'val_total_loss': total_loss.item()
        })
        return total_loss

    def test_step(self, batch: Any, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Base PyTorchLightning step for testing process.

        Args:
            batch (Any): Sample of test data.

        Returns:
            torch.Tensor: Dict with testing data distances and targets.
        """
        anchor = batch['anchor']
        anchor_label = batch['anchor_label']
        anchor_speaker = batch['anchor_speaker']
        sample = batch['sample']
        sample_label = batch['sample_label']
        sample_speaker = batch['sample_speaker']
        label_target = batch['label_target']
        speaker_target = batch['speaker_target']

        label_distances, speaker_distances = self._predict(
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

