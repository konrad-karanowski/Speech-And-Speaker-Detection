from typing import *

import hydra
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.optim import Optimizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import PYTORCH_TRANSFORMERS_CACHE

from models.siamese_model import SiameseHead, SiameseModel


class Classifier(nn.Module):

    def __init__(
        self, 
        backbone: nn.Module, 
        head_speaker: SiameseHead, 
        head_label: SiameseHead, 
        final_dim_rep: int, 
        dropout: float = 0.3) -> None:
        """Module wrapping all classifier elements.
        Uses pre-trained backbone, siamese heads and has two separate linear layers for speaker and label classification.

        Args:
            backbone (nn.Module): Pre-trained backbone.
            head_speaker (SiameseHead): Pre-trained speaker's SiameseHead.
            head_label (SiameseHead): Pre-trained label's SiameseHead.
            final_dim_rep (int): Size of final vector returned by SiameseHeads.
            dropout (float, optional): Dropout for stabilization. Defaults to 0.3.
        """
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.speaker_classifier = nn.Sequential(
            head_speaker,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim_rep, 2)
        )
        self.label_classifier = nn.Sequential(
            head_label,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim_rep, 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        fe = self.backbone(x)
        logit_speaker = self.speaker_classifier(fe)
        logit_label = self.label_classifier(fe)
        return logit_speaker, logit_label


class ClassifierModel(pl.LightningModule):

    def __init__(self, **kwargs):
        """Lightning classifier models for fine-tuning. Used as final model for inference.
        """
        super(ClassifierModel, self).__init__()
        self.save_hyperparameters()

        model = SiameseModel.load_from_checkpoint(self.hparams.checkpoint)
        self.criterion_label = nn.CrossEntropyLoss()
        self.criterion_speaker = nn.CrossEntropyLoss()

        self.classifier = Classifier(
            backbone=model.backbone,
            head_speaker=model.head_speaker,
            head_label=model.head_label,
            final_dim_rep=model.hparams.final_dim_rep,
            dropout=self.hparams.dropout
        )

        # for model inference
        self.preprocess_method = hydra.utils.instantiate(self.hparams.process_audio_method, _partial_=True)
        self.spectogram_method = hydra.utils.instantiate(self.hparams.spectrogram_method, _partial_=True)

    def configure_optimizers(self) -> Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Configure optimizer and lr scheduler.

        Returns:
            Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]: Optimizer or optimizer and lr scheduler.
        """
        params = list(self.classifier.parameters())

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
            Tuple[torch.Tensor, torch.Tensor]: Logits of speaker and label predictions.
        """
        logit_speaker, logit_label = self.classifier(x)
        return logit_speaker, logit_label

    def _predict(self,
                    query: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict whether sample belongs represents target_speaker and target_label.

        Args:
            query (torch.Tensor): Input for the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits of speaker and label predictions.
        """
        with torch.no_grad():
            logit_speaker, logit_label = self.classifier(query)
            proba_speaker = logit_speaker.softmax(dim=1)
            proba_label = logit_label.softmax(dim=1)

            return proba_speaker, proba_label

    def _process_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Processes audio samples given with request. Allows to create API request in simple way, with no hydra initialization.

        Args:
            audio (np.ndarray): Audio signal of shape (N,).
            sr (int): Audio's sampling rate.

        Returns:
            torch.Tensor: Spectrogram from audio signal.
        """
        signal = self.preprocess_method(audio, sr)
        spectrogram = self.spectogram_method(signal, self.hparams.process_audio_method.target_sr)
        spectrogram = torch.tensor(spectrogram)[None, None, :, :]
        return spectrogram

    def predict(self,
                query_samples: List[Tuple[np.ndarray, int]]
                ) -> Dict[str, np.ndarray]:
        """Predicts whether sample belongs represents target_speaker and target_label. Used for hosting the model as service.

        Args:
            query_samples (List[Tuple[np.ndarray, int]]): List of audio signals with their sample rates.

        Returns:
            Dict[str, np.ndarray]: Dictionary with probabilities of label and speaker.
        """
        query = torch.cat([self._process_audio(*sample) for sample in query_samples]).float()

        proba_speaker, proba_label = self._predict(query)
        return {
            'label_proba': proba_speaker[:, 1].cpu().numpy().tolist(),
            'speaker_proba': proba_label[:, 1].cpu().numpy().tolist(),
        }

    def training_step(self, batch: Any, *args, **kwargs) -> torch.Tensor:
        """Base PyTorchLightning step for training process.

        Args:
            batch (Any): Sample of training data.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y_speaker, y_label = batch['x'], batch['y_speaker'], batch['y_label']
        speaker_logit, label_logit = self.classifier(x)

        speaker_loss = self.criterion_speaker(speaker_logit, y_speaker)
        label_loss = self.criterion_label(label_logit, y_label)

        total_loss = self.hparams.speaker_a * speaker_loss + self.hparams.label_a * label_loss

        self.log_dict({
            'train_label_loss': label_loss.item(),
            'train_speaker_loss': speaker_loss.item(),
            'train_total_loss': total_loss.item()
        })
        return total_loss

    def validation_step(self, batch: Any, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Base PyTorchLightning step for validation process.

        Args:
            batch (Any): Sample of validation data.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, y_speaker, y_label = batch['x'], batch['y_speaker'], batch['y_label']
        speaker_logit, label_logit = self.classifier(x)

        speaker_loss = self.criterion_speaker(speaker_logit, y_speaker)
        label_loss = self.criterion_label(label_logit, y_label)

        total_loss = self.hparams.speaker_a * speaker_loss + self.hparams.label_a * label_loss

        self.log_dict({
            'val_label_loss': label_loss.item(),
            'val_speaker_loss': speaker_loss.item(),
            'val_total_loss': total_loss.item()
        })
        return {
            'speaker_logit': speaker_logit.cpu(),
            'label_logit': label_logit.cpu(),
            'speaker_target': y_speaker.cpu(),
            'label_target': y_label.cpu()
        }

    def test_step(self, batch: Any, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Base PyTorchLightning step for testing process.

        Args:
            batch (Any): Sample of testing data.

        Returns:
            torch.Tensor: Test samples predictions and targets.
        """
        x, y_speaker, y_label = batch['x'], batch['y_speaker'], batch['y_label']
        speaker_logit, label_logit = self.classifier(x)


        return {
            'speaker_logit': speaker_logit.cpu(),
            'label_logit': label_logit.cpu(),
            'speaker_target': y_speaker.cpu(),
            'label_target': y_label.cpu()
        }

    def _calculate_metrics(self, phase: str, role: str, y_true: Iterable[int], y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics from targets and predictions given by the model.

        Args:
            phase (str): Testing or validating phase.
            role (str): Dimension for whom metrics are calculated.
            y_true (Iterable[int]): True labels.
            y_pred (np.ndarray): Predictions given by the model.

        Returns:
            Dict[str, float]: Dictionary of metrics for loging.
        """
        return {
            f'{phase}_{role}_accuracy': accuracy_score(y_true, y_pred),
            f'{phase}_{role}_f1_score': f1_score(y_true, y_pred),
            f'{phase}_{role}_precision_score': precision_score(y_true, y_pred),
            f'{phase}_{role}_recall_score': recall_score(y_true, y_pred),
        }

    def _val_test_end(self, phase: str, outputs: Iterable[Dict[str, torch.Tensor]]) -> None:
        label_trues = []
        speaker_trues = []
        label_preds = []
        speaker_preds = []
        for output in outputs:
            label_trues.extend(output['label_target'])
            speaker_trues.extend(output['speaker_target'])
            label_preds.extend(output['label_logit'].argmax(dim=1))
            speaker_preds.extend(output['speaker_logit'].argmax(dim=1))
        label_dict = self._calculate_metrics(phase, 'label', label_trues, np.array(label_preds))
        speaker_dict = self._calculate_metrics(PYTORCH_TRANSFORMERS_CACHE, 'speaker', speaker_trues, np.array(speaker_preds))
        self.log_dict(label_dict)
        self.log_dict(speaker_dict)

    def validation_epoch_end(self, outputs: Iterable[Dict[str, torch.Tensor]]) -> None:
        """Calculate metrics at the end of testing process.

        Args:
            outputs (Iterable[Dict[str, torch.Tensor]]): Outputs from validation step.
        """
        self._val_test_end('val', outputs)


    def test_epoch_end(self, outputs: Iterable[Dict[str, torch.Tensor]]) -> None:
        """Calculate metrics at the end of testing process.

        Args:
            outputs (Iterable[Dict[str, torch.Tensor]]): Outputs from training step.
        """
        self._val_test_end('test', outputs)