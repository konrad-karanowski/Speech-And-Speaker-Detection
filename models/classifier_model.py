from typing import *


import hydra
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.siamese_model import SiameseHead, SiameseModel


class Classifier(nn.Module):

    def __init__(self, backbone: nn.Module, head_speaker: SiameseHead, head_label: SiameseHead, final_dim_rep: int, dropout: float = 0.3):
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
            params = list(self.classifier.parameters())

            optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=params, _convert_="partial")
            if "lr_scheduler" not in self.hparams:
                return [optimizer]
            scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.hparams.monitor}
            return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logit_speaker, logit_label = self.classifier(x)
        return logit_speaker, logit_label

    def _predict(self,
                    query: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
            
            with torch.no_grad():
                logit_speaker, logit_label = self.classifier(query)
                proba_speaker = logit_speaker.softmax(dim=1)
                proba_label = logit_label.softmax(dim=1)

                return proba_speaker, proba_label

    def _process_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        signal = self.preprocess_method(audio, sr)
        spectrogram = self.spectogram_method(signal, self.hparams.process_audio_method.target_sr)
        spectrogram = torch.tensor(spectrogram)[None, None, :, :]
        return spectrogram

    def predict(self,
                query_samples: List[Tuple[np.ndarray, int]]
                ) -> Dict[str, np.ndarray]:
        query = torch.cat([self._process_audio(*sample) for sample in query_samples]).float()

        proba_speaker, proba_label = self._predict(query)
        return {
            'label_proba': proba_speaker.cpu().numpy().tolist(),
            'speaker_proba': proba_label.cpu().numpy().tolist(),
        }

    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> torch.Tensor:
        x, y_speaker, y_label = batch['x'], batch['y_speaker'], batch['y_label']
        speaker_logit, label_logit = self.classifier(x)

        speaker_loss = self.criterion_speaker(speaker_logit, y_speaker)
        label_loss = self.criterion_label(label_logit, y_label)

        total_loss = speaker_loss + label_loss

        self.log_dict({
            'train_label_loss': label_loss.item(),
            'train_speaker_loss': speaker_loss.item(),
            'train_total_loss': total_loss.item()
        })
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> torch.Tensor:
        x, y_speaker, y_label = batch['x'], batch['y_speaker'], batch['y_label']
        speaker_logit, label_logit = self.classifier(x)

        speaker_loss = self.criterion_speaker(speaker_logit, y_speaker)
        label_loss = self.criterion_label(label_logit, y_label)

        total_loss = speaker_loss + label_loss

        self.log_dict({
            'val_label_loss': label_loss.item(),
            'val_speaker_loss': speaker_loss.item(),
            'val_total_loss': total_loss.item()
        })
        return total_loss

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> Dict[str, torch.Tensor]:
        x, y_speaker, y_label = batch['x'], batch['y_speaker'], batch['y_label']
        speaker_logit, label_logit = self.classifier(x)


        return {
            'speaker_logit': speaker_logit.cpu(),
            'label_logit': label_logit.cpu(),
            'speaker_target': y_speaker.cpu(),
            'label_target': y_label.cpu()
        }

    def _calculate_metrics(self, role: str, y_true: Iterable[int], y_pred: np.ndarray) -> Dict[str, float]:
        return {
            f'{role}_accuracy': accuracy_score(y_true, y_pred),
            f'{role}_f1_score': f1_score(y_true, y_pred),
            f'{role}_precision_score': precision_score(y_true, y_pred),
            f'{role}_recall_score': recall_score(y_true, y_pred),
        }

    def test_epoch_end(self, outputs: Iterable[Dict[str, torch.Tensor]]) -> None:

        label_trues = []
        speaker_trues = []
        label_preds = []
        speaker_preds = []
        for output in outputs:
            label_trues.extend(output['label_target'])
            speaker_trues.extend(output['speaker_target'])
            label_preds.extend(output['label_logit'].argmax(dim=1))
            speaker_preds.extend(output['speaker_logit'].argmax(dim=1))
        label_dict = self._calculate_metrics('label', label_trues, np.array(label_preds))
        speaker_dict = self._calculate_metrics('speaker', speaker_trues, np.array(speaker_preds))
        self.log_dict(label_dict)
        self.log_dict(speaker_dict)