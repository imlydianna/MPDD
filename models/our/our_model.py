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

'''
class ourModel(BaseModel, nn.Module):
    def __init__(self, opt):
        """
        Αρχικοποίηση του μοντέλου.
        opt: Αντικείμενο με όλες τις παραμέτρους (από config.json και command-line)
        """
        super(ourModel, self).__init__() # Καλεί τον constructor και των δύο γονικών κλάσεων
        self.opt = opt
        self.model_names = [] # Λίστα για την αποθήκευση των sub-models

        # --- ΟΡΙΣΜΟΣ ΥΠΟ-ΜΟΝΤΕΛΩΝ ---
        
        # 1. Acoustic Model (LSTM για ήχο)
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('EmoA')

        # 2. Visual Model (LSTM για βίντεο)
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('EmoV')
        
        # 3. Personalized Features Model (Ένα απλό δίκτυο για τα δημογραφικά/κείμενο)
        # Μετατρέπει τα 768 features σε μια πιο διαχειρίσιμη διάσταση (π.χ., 128)
        self.netEmoP = nn.Sequential(
            nn.Linear(opt.input_dim_p, 128), # input_dim_p θα οριστεί στο train.py
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.model_names.append('EmoP')

        # 4. Transformer Fusion Model
        # Η διάσταση εισόδου στον Transformer
        transformer_input_dim = opt.embd_size_a + opt.embd_size_v
        self.transformer_projector = nn.Linear(transformer_input_dim, opt.hidden_size) # Προβολή στην hidden_size του Transformer
        
        emo_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head), batch_first=True)
        self.netEmoFusion = nn.TransformerEncoder(encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # 5. Τελικός Ταξινομητής (Classifier)
        cls_layers = [int(x) for x in opt.cls_layers.split(',')]
        
        # Η είσοδος στον classifier είναι το "flattened" output του Transformer + το output του Personalized model
        cls_input_size = opt.feature_max_len * opt.hidden_size + 128 # 128 από το netEmoP
                                            
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')

        # --- ΟΡΙΣΜΟΣ LOSS FUNCTIONS & OPTIMIZER ---
        self.loss_names = ['emo_CE', 'focal'] # Τα ονόματα των losses που θα καταγράφουμε
        
        if self.isTrain:
            # Οι loss functions θα αρχικοποιηθούν σωστά στο train.py με τα class weights
            self.criterion_ce = torch.nn.CrossEntropyLoss() 
            self.criterion_focal = Focal_Loss(gamma=opt.focal_gamma)
            
            # Ο optimizer θα βελτιστοποιεί όλες τις παραμέτρους όλων των sub-models
            parameters = []
            for net_name in self.model_names:
                net = getattr(self, 'net' + net_name)
                parameters += list(net.parameters())
                
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers.append(self.optimizer)
            
            # Βάρη για τη συνολική loss
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight


    def set_input(self, input):
        """
        Αποθηκεύει τα δεδομένα από τον dataloader και τα μεταφέρει στη σωστή συσκευή.
        """
        self.acoustic = input['audio_feature'].float().to(self.device)
        self.visual = input['video_feature'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        
        # Χειρισμός των personalized features
        if 'individual_feature' in input:
            self.personalized = input['individual_feature'].float().to(self.device)
        else:
            self.personalized = None


    def forward(self):
        """
        Η κύρια λογική του μοντέλου (forward pass).
        """
        batch_size = self.acoustic.size(0)

        # 1. Παίρνουμε τα embeddings από ήχο και βίντεο μέσω των LSTMs
        emo_feat_A = self.netEmoA(self.acoustic)  # (B, L, embd_size_a)
        emo_feat_V = self.netEmoV(self.visual)    # (B, L, embd_size_v)

        # 2. Ενώνουμε τα audio/video features και τα προβάλλουμε στη διάσταση του Transformer
        multimodal_feat_AV = torch.cat((emo_feat_A, emo_feat_V), dim=-1) # (B, L, embd_size_a + embd_size_v)
        transformer_in = self.transformer_projector(multimodal_feat_AV) # (B, L, hidden_size)

        # 3. Περνάμε την ενωμένη ακολουθία από τον Transformer
        emo_fusion_feat = self.netEmoFusion(transformer_in) # (B, L, hidden_size)

        # 4. Κάνουμε "flatten" το output του Transformer
        emo_fusion_flat = emo_fusion_feat.reshape(batch_size, -1)  # (B, L * hidden_size)

        # 5. Περνάμε τα personalized features από το δικό τους δίκτυο
        if self.personalized is not None:
            emo_feat_P = self.netEmoP(self.personalized) # (B, 128)
            
            # 6. Τελική συνένωση (fusion) όλων των χαρακτηριστικών
            final_fusion_feat = torch.cat((emo_fusion_flat, emo_feat_P), dim=-1)
        else:
            # Αν δεν υπάρχουν personalized features (π.χ. για debugging)
            # ΠΡΟΣΟΧΗ: Αυτό θα δώσει error αν ο classifier περιμένει άλλη διάσταση.
            # Πρέπει να διασφαλίσουμε ότι το personalized feature υπάρχει πάντα.
            final_fusion_feat = emo_fusion_flat

        # 7. Τελική ταξινόμηση
        self.emo_logits, _ = self.netEmoC(final_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)


    def backward(self):
        """
        Υπολογισμός της συνολικής loss.
        """
        # --- Cross-Entropy Loss ---
        # Ειδικός χειρισμός για soft labels (από Mixup)
        if self.emo_label.dtype == torch.float32:
            loss_ce = -torch.sum(F.log_softmax(self.emo_logits, dim=-1) * self.emo_label, dim=-1).mean()
        else: # Κανονική CE για hard labels
            loss_ce = self.criterion_ce(self.emo_logits, self.emo_label)
        
        self.loss_emo_CE = self.ce_weight * loss_ce
        
        # --- Focal Loss ---
        # Η βελτιωμένη Focal Loss θα επιστρέψει 0 αν οι ετικέτες είναι soft.
        self.loss_focal = self.focal_weight * self.criterion_focal(self.emo_logits, self.emo_label)

        # --- Συνολική Loss ---
        total_loss = self.loss_emo_CE + self.loss_focal
        total_loss.backward()


    def optimize_parameters(self, epoch):
        """
        Εκτελεί ένα πλήρες βήμα εκπαίδευσης: forward, backward, update weights.
        """
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) # Gradient clipping για σταθερότητα
        self.optimizer.step()

    def get_current_losses(self):
        # Επιστρέφει τα losses για logging
        return OrderedDict([('emo_CE', self.loss_emo_CE.item()), ('focal', self.loss_focal.item())])

    def test(self):
        """
        Λειτουργία αξιολόγησης (inference).
        """
        with torch.no_grad():
            self.forward()
            
'''
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

        #insure time dimension modification
        emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1) # (batch_size, seq_len, 2 * embd_size)
        
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        
        #dynamic acquisition of bs
        batch_size = emo_fusion_feat.size(0)

        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)  # turn into [batch_size, feature_dim] 1028

        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)  # [batch_size, seq_len * feature_dim + 1024]

        #for back prop
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


# FOCAL LOSS - Με βελτιώσεις για να δουλεύει σωστά με class weights και mixup

class Focal_Loss(nn.Module):
    """
    Βελτιωμένη Focal Loss που χειρίζεται σωστά τα class weights και αγνοείται 
    όταν οι ετικέτες είναι "soft" (π.χ. από Mixup).
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Το alpha είναι ο tensor με τα βάρη των κλάσεων
        self.alpha = alpha

    def forward(self, preds, targets):
        # Αν οι ετικέτες δεν είναι ακέραιοι αριθμοί (δηλαδή είναι soft από Mixup),
        # τότε η Focal Loss δεν μπορεί να εφαρμοστεί. Επιστρέφουμε loss ίσο με 0.
        if targets.dtype != torch.long:
            return torch.tensor(0.0, device=preds.device)

        # Υπολογίζουμε την CrossEntropyLoss ανά δείγμα
        ce_loss = F.cross_entropy(preds, targets, reduction='none', weight=self.alpha)
        
        # pt είναι η πιθανότητα που έδωσε το μοντέλο στη σωστή κλάση
        pt = torch.exp(-ce_loss)
        
        # Ο όρος της Focal Loss
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
