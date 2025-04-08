import json
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
from collections import Counter

class AudioVisualDataset(Dataset):
    def __init__(self, json_data, label_count, personalized_feature_file, max_len=10, batch_size=32,
                 audio_path='', video_path='', isTest=False, augment=True):
        """
        Dataset class με ενσωματωμένο oversampling και data augmentation για rare classes.
        """
        self.data = json_data
        self.max_len = max_len
        self.batch_size = batch_size
        self.isTest = isTest
        self.augment = augment
        self.label_count = label_count
        self.audio_path = audio_path
        self.video_path = video_path

        # === Load personalized embeddings ===
        self.personalized_features = self.load_personalized_features(personalized_feature_file)

        if not isTest:
            # === Επιλογή label field ανάλογα με το task ===
            self.label_field = {2: "bin_category", 3: "tri_category", 5: "pen_category"}[label_count]

            # === Μέτρησε κατανομή labels ===
            self.label_freq = Counter([entry[self.label_field] for entry in self.data])

            # === Oversample classes με λιγότερα από 100 δείγματα ===
            self.minority_classes = [label for label, count in self.label_freq.items() if count < 100]
        else:
            self.minority_classes = []

    def __len__(self):
        return len(self.data)

    def load_personalized_features(self, file_path):
        """Load personalized feature vectors from .npy file."""
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and isinstance(data[0], dict):
            return {entry["id"]: entry["embedding"] for entry in data}
        else:
            raise ValueError("❗ Unexpected .npy structure — expected list of dicts.")

    def pad_or_truncate(self, feature, max_len):
        """Pad or truncate to fixed length (max_len)."""
        if feature.shape[0] < max_len:
            padding = torch.zeros((max_len - feature.shape[0], feature.shape[1]))
            feature = torch.cat((feature, padding), dim=0)
        else:
            feature = feature[:max_len]
        return feature

    def augment_feature(self, feature, noise_std=0.01, dropout_p=0.1):
        """
        Εφαρμογή data augmentation:
        - Gaussian noise
        - Dropout-style masking
        """
        if self.augment:
            noise = torch.randn_like(feature) * noise_std
            feature += noise
            mask = (torch.rand_like(feature[:, :1]) > dropout_p).float()
            feature = feature * mask
        return feature

    def __getitem__(self, idx):
        entry = self.data[idx]

        # === Φόρτωσε τα audio/video features ===
        audio = np.load(os.path.join(self.audio_path, entry['audio_feature_path']))
        video = np.load(os.path.join(self.video_path, entry['video_feature_path']))
        audio = torch.tensor(audio, dtype=torch.float32)
        video = torch.tensor(video, dtype=torch.float32)

        # === Pad/Truncate ===
        audio = self.pad_or_truncate(audio, self.max_len)
        video = self.pad_or_truncate(video, self.max_len)

        # === Label ===
        if self.isTest:
            label = 0
        else:
            label = torch.tensor(entry[self.label_field], dtype=torch.long)

        # === Oversampling Target ===
        is_rare = label.item() in self.minority_classes
        if is_rare:
            audio = self.augment_feature(audio)
            video = self.augment_feature(video)

        # === Personalized Embedding ===
        person_id = str(int(os.path.basename(entry['audio_feature_path']).split('_')[0]))
        if person_id in self.personalized_features:
            pers_feat = torch.tensor(self.personalized_features[person_id], dtype=torch.float32)
        else:
            pers_feat = torch.zeros(1024)
            print(f"❗ Personalized embedding not found for id {person_id}")

        return {
            'A_feat': audio,
            'V_feat': video,
            'emo_label': label,
            'personalized_feat': pers_feat
        }
