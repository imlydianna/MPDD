from datetime import datetime
import os
import json
import time
import argparse
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from train_val_split import train_val_split1, train_val_split2
from models.our.our_model import ourModel
#from dataset import *
from custom_dataset import AudioVisualDataset  # το custom_dataset.py
from utils.logger import get_logger
import numpy as np
from collections import Counter # νέο 
from torch.utils.data import WeightedRandomSampler # νέο
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight



class Opt:
    """Helper class για τη φόρτωση του config.json"""
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    """Φορτώνει το configuration από ένα JSON αρχείο."""
    with open(config_file, 'r') as f:
        return json.load(f)

def eval(model, val_loader, device):
    """Συνάρτηση αξιολόγησης του μοντέλου στο validation set."""
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for data in val_loader:
            # Μεταφορά δεδομένων στη συσκευή (CPU/GPU)
            for k, v in data.items():
                data[k] = v.to(device)
                
            model.set_input(data)
            model.test()  # Κάνει forward pass χωρίς υπολογισμό gradient

            # Συλλογή προβλέψεων και πραγματικών ετικετών
            preds = model.emo_pred.argmax(dim=1).cpu().numpy()
            labels = data['emo_label'].cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels)

    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)

    # Υπολογισμός μετρικών
    acc_weighted = accuracy_score(total_labels, total_preds, sample_weight=None) # Το 'weighted' εδώ είναι το default του sklearn
    acc_unweighted = accuracy_score(total_labels, total_preds) # Είναι το ίδιο με το macro average

    f1_weighted = f1_score(total_labels, total_preds, average='weighted')
    f1_unweighted = f1_score(total_labels, total_preds, average='macro')
    cm = confusion_matrix(total_labels, total_preds)

    return acc_weighted, acc_unweighted, f1_weighted, f1_unweighted, cm


def train_model(train_json, model, audio_path, video_path, max_len,
                best_model_name, seed, class_weights_tensor=None):  # νέο όρισμα class_weights_tensor=None
    """
    This is the training function - κύρια συνάρτηση εκπαίδευσης του μοντέλου
    """
    logger.info(f'personalized features used：{args.personalized_features_file}')
    num_epochs = args.num_epochs
    device = args.device
    print(f"device: {device}")
    model.to(device)
    print("Training device:", device)
                    
    #scheduler = CosineAnnealingLR(model.optimizer, T_max=num_epochs, eta_min=1e-6)
    # Reduce LR if no improvement
    scheduler = ReduceLROnPlateau(model.optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Early Stopping variables
    early_stop_counter = 0
    early_stop_patience = 20  # μπορώ να το προσαρμόσω



    # προσθήκη Class Weights στην CrossEntropyLoss                 
    model.criterion_ce = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))


    # split training and validation set
    # data = json.load(open(train_json, 'r'))
    if args.track_option=='Track1':
        train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
    elif args.track_option=='Track2':
        train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=0.1,
                                                                                     seed=seed)
    
    # Δημιουργία Datasets και DataLoaders
    # Mixup=True για data augmentation στο training set
    train_dataset = AudioVisualDataset(train_data, args.labelcount, args.personalized_features_file, max_len,
                                       batch_size=args.batch_size,
                                       audio_path=audio_path, video_path=video_path, use_mixup=True, mixup_alpha=0.4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

                    
    #  Mixup=False για το validation set
    val_dataset = AudioVisualDataset(val_data, args.labelcount, args.personalized_features_file, max_len,
                                     batch_size=args.batch_size,
                                     audio_path=audio_path, video_path=video_path, use_mixup=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    #logger.info('The number of training samples = %d' % len(train_loader.dataset))
    #logger.info('The number of val samples = %d' % len(val_loader.dataset))

    logger.info(f'Number of training samples = {len(train_dataset)}')
    logger.info(f'Number of validation samples = {len(val_dataset)}')

    best_f1_weighted = 0.0
    best_f1_unweighted = 0.0
    best_acc_weighted = 0.0
    best_acc_unweighted = 0.0
    best_epoch = 0
    best_cm = None

    for epoch in range(num_epochs):
        model.train(True)
        total_loss = 0

        for i, data in enumerate(train_loader):
            for k, v in data.items():
                data[k] = v.to(device)
            model.set_input(data)
            model.optimize_parameters(epoch)

            losses = model.get_current_losses()
            total_loss += losses['emo_CE']

        avg_loss = total_loss / len(train_loader)

        # evaluation
        acc_w, acc_uw, f1_w, f1_uw, cm = eval(model, val_loader, device)
        
        # Step scheduler with current validation metric
        scheduler.step(f1_w)  #  παρακολουθούμε το emo_f1_weighted ως βασικό metric για ReduceLROnPlateau

        current_lr = model.optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.8f}")
 

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}, LR: {current_lr:.8f}")
        logger.info(f"Val -> Weighted F1: {f1_w:.4f}, Unweighted F1: {f1_uw:.4f}, Weighted Acc: {acc_w:.4f}, Unweighted Acc: {acc_uw:.4f}")
        logger.info('Confusion Matrix:\n{}'.format(cm))
        
        # Έλεγχος για αποθήκευση του καλύτερου μοντέλου και Early Stopping
        if f1_w > best_f1_weighted:
            logger.info("New best model found! Saving model state...")
            best_f1_weighted = f1_w
            best_f1_unweighted = f1_uw
            best_acc_weighted = acc_w
            best_acc_unweighted = acc_uw
            best_cm = cm
            best_epoch = epoch + 1
            early_stop_counter = 0  # Reset τον counter
            
            save_path = os.path.join(opt.checkpoints_dir, opt.name, best_model_name)
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1
            logger.info(f"No improvement. Early Stop Counter: {early_stop_counter}/{early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            logger.info("Early stopping triggered.")
            break
            

    # Τελικά αποτελέσματα
    logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_epoch}.")
    logger.info(f"Best Weighted F1: {best_f1_weighted:.4f}, Best Unweighted F1: {best_f1_unweighted:.4f}, "
                f"Best Weighted Acc: {best_acc_weighted:.4f}, Best Unweighted Acc: {best_acc_unweighted:.4f}.")
    logger.info('Best Confusion Matrix:\n{}'.format(best_cm))

    # Αποθήκευση αποτελεσμάτων σε CSV
    csv_file = f'{opt.log_dir}/{opt.name}_results.csv'
    formatted_best_cm = '"' + str(best_cm.tolist()).replace('\n', '') + '"'
    header = "Time,Seed,SplitWindow,Labels,Audio,Video,BatchSize,Epochs,MaxLen,LR,WeightedF1,UnweightedF1,WeightedAcc,UnweightedAcc,ConfusionMatrix"
    result_line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{args.splitwindow_time},{args.labelcount},"
                   f"{args.audiofeature_method},{args.videofeature_method},{args.batch_size},{best_epoch},"
                   f"{opt.feature_max_len},{opt.lr:.6f},{best_f1_weighted:.4f},{best_f1_unweighted:.4f},"
                   f"{best_acc_weighted:.4f},{best_acc_unweighted:.4f},{formatted_best_cm}")

    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as file:
            file.write(header + '\n')
    
    with open(csv_file, 'a') as file:
        file.write(result_line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MPDD Model")
    # ... (Όλες οι γραμμές του argparse παραμένουν ακριβώς οι ίδιες όπως τις είχες)
    parser.add_argument('--labelcount', type=int, default=3, help="Number of data categories (2, 3, or 5).")
    parser.add_argument('--track_option', type=str, required=True, help="Track1 or Track2")
    parser.add_argument('--feature_max_len', type=int, required=True, help="Max length of feature.")
    parser.add_argument('--data_rootpath', type=str, required=True, help="Root path to the program dataset")
    parser.add_argument('--train_json', type=str, required=False, help="File name of the training JSON file")
    parser.add_argument('--personalized_features_file', type=str, help="File name of the personalized features file")
    parser.add_argument('--audiofeature_method', type=str, default='mfccs', choices=['mfccs', 'opensmile', 'wav2vec'], help="Method for extracting audio features.")
    parser.add_argument('--videofeature_method', type=str, default='densenet', choices=['openface', 'resnet', 'densenet'], help="Method for extracting video features.")
    parser.add_argument('--splitwindow_time', type=str, default='1s', help="Time window for splitted features. e.g. '1s' or '5s'")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs to train the model")
    parser.add_argument('--device', type=str, default='cuda', help="Device to train the model on, e.g. 'cuda' or 'cpu'")
    
    args = parser.parse_args()

    # --- SETUP & CONFIGURATION ---
    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = load_config('config.json')
    opt = Opt(config)
    opt.emo_output_dim = args.labelcount
    opt.feature_max_len = args.feature_max_len
    opt.lr = args.lr

    # Paths
    args.train_json = os.path.join(args.data_rootpath, 'Training', 'labels', 'Training_Validation_files.json')
    args.personalized_features_file = os.path.join(args.data_rootpath, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')
    audio_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'
    video_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'

    # Logger
    opt.name = f'{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}'
    logger_path = os.path.join(opt.log_dir, opt.name)
    os.makedirs(logger_path, exist_ok=True)
    logger = get_logger(logger_path, 'result')

    # --- CLASS WEIGHTING ---
    if args.track_option == 'Track1':
        train_data, _, _, _ = train_val_split1(args.train_json, val_ratio=0.1, random_seed=seed)
    else:
        train_data, _, _, _ = train_val_split2(args.train_json, val_percentage=0.1, seed=seed)
    
    label_key = {2: "bin_category", 3: "tri_category", 5: "pen_category"}[args.labelcount]
    train_labels = [sample[label_key] for sample in train_data]
    
    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    # --- MODEL INITIALIZATION ---
    # Obtain input_dim_a, input_dim_v
    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(os.path.join(audio_path, filename)).shape[1]
            break
    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(os.path.join(video_path, filename)).shape[1]            
            break
            
    opt.input_dim_p = 768  # Διάσταση του embedding από το personalized features file

    model = ourModel(opt)

    # --- LOGGING & TRAINING ---
    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    best_model_name = f"best_model_{cur_time}.pth"

    logger.info(f"Starting training run: {opt.name}")
    logger.info(f"Parameters: splitwindow_time={args.splitwindow_time}, audio={args.audiofeature_method}, video={args.videofeature_method}")
    logger.info(f"batch_size={args.batch_size}, epochs={args.num_epochs}, lr={opt.lr}")
    logger.info(f"Using random seed: {seed}")

    train_model(
        train_json_path=args.train_json,
        model=model,
        max_len=opt.feature_max_len,
        best_model_name=best_model_name,
        audio_path=audio_path,
        video_path=video_path,
        seed=seed,
        class_weights_tensor=class_weights
    )
