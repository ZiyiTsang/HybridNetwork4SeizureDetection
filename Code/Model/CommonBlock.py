from typing import Any

import transformers
from torch import nn
import torch
import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision


class Classification_block(nn.Module):
    def __init__(self):
        super(Classification_block, self).__init__()
        self.linear_1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear_2(self.dropout(self.relu(self.linear_1(x))))).squeeze()

class CombineEncoder(nn.Module):
    def __init__(self, encoder1,encoder2):
        super(CombineEncoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

        self.encoder_name = "CombineEncoder"


    def forward(self, x):
        x1=self.encoder1(x)
        x2=self.encoder2(x)
        return (x1+x2)/2

class CommonNet(L.LightningModule):
    def __init__(self, encoder, lr=1e-4):
        super().__init__()
        self.encoder = encoder
        self.classification = Classification_block()
        # ---
        self.binary_accuracy = BinaryAccuracy()
        self.binary_f1 = BinaryF1Score()
        self.AUC = BinaryAUROC(thresholds=60)
        self.AveragePrecision = BinaryAveragePrecision(thresholds=60)
        # ---
        self.lr = lr
        self.save_hyperparameters(ignore=['encoder'])
    def forward(self,X) -> Any:
        return self.classification(self.encoder(X)).view(-1)

    def training_step(self, batch, batch_idx):
        X, y = batch

        y_hat = self.forward(X)
        # loss = sigmoid_focal_loss(y_hat, y, reduction='mean')
        loss = nn.BCELoss()(y_hat, y)
        # acc = self.binary_accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        acc = self.binary_accuracy(y_hat, y)
        f1 = self.binary_f1(y_hat, y)
        # loss = sigmoid_focal_loss(y_hat, y, reduction='mean')
        loss = nn.BCELoss()(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y=y.int()
        y_hat = self.forward(X)
        acc = self.binary_accuracy(y_hat, y)
        f1 = self.binary_f1(y_hat, y)
        auc = self.AUC(y_hat, y)
        ap = self.AveragePrecision(y_hat, y)
        loss = nn.BCELoss()(y_hat, y.float())
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)
        self.log('test_f1', f1, on_step=True, on_epoch=True)
        self.log('test_auc', auc, on_step=True, on_epoch=True)
        self.log('test_ap', ap, on_step=True, on_epoch=True)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)
        if self.logger is not None:
            self.logger.experiment.log_metric('lr', scheduler.get_last_lr()[0], step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.encoder.encoder_name=='transformer' or "CombineEncoder":
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5,
                                                                        num_training_steps=self.trainer.estimated_stepping_batches)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
        return [optimizer], [lr_scheduler]
