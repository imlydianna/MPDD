import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
import math
import torch.nn as nn


class ourModel(BaseModel, nn.Module):

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        nn.Module.__init__(self)
        super().__init__(opt)

        
        self.loss_names = []
        self.model_names = []

        # acoustic model
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('EmoA')

        # visual model
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('EmoV')

        # Transformer Fusion model
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head), batch_first=True)
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # cls_input_size = 5*opt.hidden_size, same with max-len
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024  # with personalized feature

                                            
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.temperature = opt.temperature


        # self.device = 'cpu'
        # self.netEmoA = self.netEmoA.to(self.device)
        # self.netEmoV = self.netEmoV.to(self.device)
        # self.netEmoFusion = self.netEmoFusion.to(self.device)
        # self.netEmoC = self.netEmoC.to(self.device)
        # self.netEmoCF = self.netEmoCF.to(self.device)

        self.criterion_ce = torch.nn.CrossEntropyLoss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss() 
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        # modify save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  


    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):

        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)

        self.emo_label = input['emo_label'].to(self.device)

        if 'personalized_feat' in input:
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None  # if no personalized features given
            

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        emo_feat_A = self.netEmoA(self.acoustic)
        emo_feat_V = self.netEmoV(self.visual)

        '''insure time dimension modification'''
        emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1) # (batch_size, seq_len, 2 * embd_size)
        
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        
        '''dynamic acquisition of bs'''
        batch_size = emo_fusion_feat.size(0)

        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)  # turn into [batch_size, feature_dim] 1028

        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)  # [batch_size, seq_len * feature_dim + 1024]

        '''for back prop'''
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        """-----------"""

        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        """Calculate the loss for back propagation"""
        #self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label) 
        # Αν οι ετικέτες είναι soft (από Mixup) - CE loss: Mixup-aware
        if self.emo_label.dtype == torch.float32:
            # Χρησιμοποιούμε soft cross-entropy
            self.loss_emo_CE = torch.sum(-self.emo_label * F.log_softmax(self.emo_logits, dim=-1), dim=-1).mean()
        else:
            self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)

        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE

        loss.backward()

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()

        self.optimizer.step()


class ActivateFun(torch.nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


'''
class Focal_Loss(torch.nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds:softmax output
        labels:true values
        """
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")
'''
'''
class Focal_Loss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha  # tensor με class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')  # per-sample loss
        pt = torch.exp(-ce_loss)  # prob of correct class

        if self.alpha is not None:
            at = self.alpha[targets]  # παίρνεις weight για κάθε δείγμα ανάλογα με το target
        else:
            at = 1.0

        focal_loss = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise NotImplementedError("Invalid reduction mode.")
    '''
    class Focal_Loss(torch.nn.Module):
        def __init__(self, alpha=None, gamma=2, reduction='mean'):
            super(Focal_Loss, self).__init__()
            self.alpha = alpha  # tensor με class weights
            self.gamma = gamma
            self.reduction = reduction
    
        def forward(self, preds, targets):
            # Αν targets είναι soft labels (Mixup), παράκαμψε focal loss
            if targets.dtype != torch.long and targets.dtype != torch.int:
                # προαιρετικό logging
                print("⚠️ Skipping Focal Loss: received soft labels from Mixup.")
                return torch.tensor(0.0, device=preds.device, requires_grad=True)
        
            ce_loss = F.cross_entropy(preds, targets, reduction='none')
            pt = torch.exp(-ce_loss)
        
            if self.alpha is not None:
                at = self.alpha[targets]  # indexing με ακέραιους δείκτες
            else:
                at = 1.0
        
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        
            if self.reduction == 'none':
                return focal_loss
            elif self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                raise NotImplementedError("Invalid reduction mode.")
        
