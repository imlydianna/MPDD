import torch
import os
import json
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
import numpy as np


class customModel(BaseModel, nn.Module):
    def __init__(self, opt):
        super().__init__(opt)
        nn.Module.__init__(self)

        self.loss_names = []
        self.model_names = []

        # === Acoustic & Visual LSTM Encoders ===
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, embd_method=opt.embd_method_v)
        self.model_names += ['EmoA', 'EmoV']

        # === Transformer Fusion Layer ===
        emo_encoder_layer = nn.TransformerEncoderLayer(
            d_model=opt.hidden_size, nhead=int(opt.Transformer_head), dropout=opt.attention_dropout, batch_first=True
        )
        self.netEmoFusion = nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # === Classifier ===
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024  # fusion + personalized
        cls_layers = list(map(int, opt.cls_layers.split(',')))
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.temperature = opt.temperature

        # === Loss Functions ===
        if self.isTrain:
            self.criterion_ce = nn.CrossEntropyLoss()
            self.criterion_focal = Focal_Loss()

            # === Optimizer ===
            parameters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        self.personalized = input.get('personalized_feat', torch.zeros(self.acoustic.size(0), 1024)).float().to(self.device)

        # === Mixup augmentation (optional) ===
        if self.isTrain and hasattr(self.opt, 'use_mixup') and self.opt.use_mixup:
            self.acoustic, self.visual, self.emo_label = apply_mixup(self.acoustic, self.visual, self.emo_label)

    def forward(self):
        emo_feat_A = self.netEmoA(self.acoustic)
        emo_feat_V = self.netEmoV(self.visual)
        fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1)
        fusion_feat = self.netEmoFusion(fusion_feat)
        fusion_feat = fusion_feat.permute(1, 0, 2).reshape(fusion_feat.size(0), -1)
        fusion_feat = torch.cat((fusion_feat, self.personalized), dim=-1)
        self.emo_logits, _ = self.netEmoC(fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        loss_ce = self.criterion_ce(self.emo_logits, self.emo_label)
        loss_focal = self.criterion_focal(self.emo_logits, self.emo_label)
        total_loss = self.ce_weight * loss_ce + self.focal_weight * loss_focal
        total_loss.backward()

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


class Focal_Loss(nn.Module):
    def __init__(self, weight=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


def apply_mixup(audio, visual, labels, alpha=0.4):
    if alpha <= 0:
        return audio, visual, labels
    lam = np.random.beta(alpha, alpha)
    batch_size = audio.size(0)
    index = torch.randperm(batch_size)

    audio_mix = lam * audio + (1 - lam) * audio[index, :]
    visual_mix = lam * visual + (1 - lam) * visual[index, :]
    labels_mix = lam * labels + (1 - lam) * labels[index]

    return audio_mix, visual_mix, labels_mix.long()
