import json
import torch
import numpy as np
from torch.utils.data import Dataset
import os


class AudioVisualDataset(Dataset):
    #def __init__(self, json_data, label_count, personalized_feature_file, max_len=10, batch_size=32, audio_path='', video_path='', isTest=False):
    def __init__(self, json_data, label_count, personalized_feature_file, max_len=10,
             batch_size=32, audio_path='', video_path='', isTest=False, use_mixup=False, mixup_alpha=0.4):

        self.data = json_data
        self.max_len = max_len  # Expected sequence length
        self.batch_size = batch_size 
        self.isTest = isTest

        # mix-up flags
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha


        # Load personalized features
        self.personalized_features = self.load_personalized_features(personalized_feature_file)
        self.audio_path = audio_path
        self.video_path = video_path
        self.label_count = label_count

    def __len__(self):
        return len(self.data)

    def fixed_windows(self, features: torch.Tensor, fixLen=4):
        """
        Divides 2D features into fixLen fixed windows and aggregates them into fixed-size results (Tensor version).

        Parameters.
        - features: the input feature tensor of (timesteps, feature_dim)

        Returns.
        - Tensor of (4, feature_dim), each row represents a window of aggregated features
        """
        timesteps, feature_dim = features.shape
        window_size = int(torch.ceil(torch.tensor(timesteps / fixLen)))
        windows = []
        for i in range(fixLen):
            start = i * window_size
            end = min(start + window_size, timesteps)
            window = features[start:end]
            if window.size(0) > 0:
                window_aggregated = torch.mean(window, dim=0)
                windows.append(window_aggregated)
            else:
                windows.append(torch.zeros(feature_dim))

        return torch.stack(windows, dim=0)

    def pad_or_truncate(self, feature, max_len):
        """Fill or truncate the input feature sequence"""
        if feature.shape[0] < max_len:
            padding = torch.zeros((max_len - feature.shape[0], feature.shape[1]))
            feature = torch.cat((feature, padding), dim=0)
        else:
            feature = feature[:max_len]
        return feature

    def load_personalized_features(self, file_path):
        """
        Load personalized features from the .npy file.
        """

        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and isinstance(data[0], dict):
            return {entry["id"]: entry["embedding"] for entry in data}
        else:
            raise ValueError("Unexpected data format in the .npy file. Ensure it contains a list of dictionaries.")

    '''
    def __getitem__(self, idx):
        entry = self.data[idx]

        # Load audio and video features
        audio_feature = np.load(self.audio_path + '/' + entry['audio_feature_path'])
        video_feature = np.load(self.video_path + '/' + entry['video_feature_path'])
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
        video_feature = torch.tensor(video_feature, dtype=torch.float32)

        audio_feature = self.pad_or_truncate(audio_feature, self.max_len)
        video_feature = self.pad_or_truncate(video_feature, self.max_len)

        # Load label
        if self.isTest == False:
            if self.label_count == 2:
                label = torch.tensor(entry['bin_category'], dtype=torch.long)
            elif self.label_count == 3:
                label = torch.tensor(entry['tri_category'], dtype=torch.long)
            elif self.label_count == 5:
                label = torch.tensor(entry['pen_category'], dtype=torch.long)
        else:
            label = 0

        import os

        filepath = entry['audio_feature_path']  # the filename containing path to features
        filename = os.path.basename(filepath)
        # Extract person ids and convert to integers
        person_id = int(filename.split('_')[0])
        personalized_id = str(person_id)

        if personalized_id in self.personalized_features:
            personalized_feature = torch.tensor(self.personalized_features[personalized_id], dtype=torch.float32)
        else:
            # If no personalized feature found, use a zero vector
            personalized_feature = torch.zeros(1024, dtype=torch.float32)
            print(f"❗Personalized feature not found for id: {personalized_id}")

        return {
            'A_feat': audio_feature,
            'V_feat': video_feature,
            'emo_label': label,
            'personalized_feat': personalized_feature
        }
    '''
    def __getitem__(self, idx):
        def get_single_item(i):
            entry = self.data[i]

            audio_feature = np.load(self.audio_path + '/' + entry['audio_feature_path'])
            video_feature = np.load(self.video_path + '/' + entry['video_feature_path'])
            audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
            video_feature = torch.tensor(video_feature, dtype=torch.float32)

            audio_feature = self.pad_or_truncate(audio_feature, self.max_len)
            video_feature = self.pad_or_truncate(video_feature, self.max_len)

            if self.isTest == False:
                if self.label_count == 2:
                    label = entry['bin_category']
                elif self.label_count == 3:
                    label = entry['tri_category']
                elif self.label_count == 5:
                    label = entry['pen_category']
            else:
                label = 0

            filepath = entry['audio_feature_path']
            filename = os.path.basename(filepath)
            person_id = int(filename.split('_')[0])
            personalized_id = str(person_id)

            if personalized_id in self.personalized_features:
                personalized_feature = torch.tensor(self.personalized_features[personalized_id], dtype=torch.float32)
            else:
                personalized_feature = torch.zeros(1024, dtype=torch.float32)
                print(f"❗Personalized feature not found for id: {personalized_id}")

            return audio_feature, video_feature, personalized_feature, label

        if self.use_mixup and not self.isTest:
            # Επιλογή δεύτερου δείγματος
            idx2 = np.random.randint(0, len(self.data))
            a1, v1, p1, y1 = get_single_item(idx)
            a2, v2, p2, y2 = get_single_item(idx2)

            # δείκτες σε one-hot
            y1_onehot = torch.nn.functional.one_hot(torch.tensor(y1), num_classes=self.label_count).float()
            y2_onehot = torch.nn.functional.one_hot(torch.tensor(y2), num_classes=self.label_count).float()
    
            # lambda ~ Beta(α, α)
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
    
            # Mixup
            audio_mix = lam * a1 + (1 - lam) * a2
            video_mix = lam * v1 + (1 - lam) * v2
            personalized_mix = lam * p1 + (1 - lam) * p2
            label_mix = lam * y1_onehot + (1 - lam) * y2_onehot
    
            return {
                'A_feat': audio_mix,
                'V_feat': video_mix,
                'emo_label': label_mix,
                'personalized_feat': personalized_mix
            }
    
        else:
            a, v, p, y = get_single_item(idx)
            return {
                'A_feat': a,
                'V_feat': v,
                'emo_label': torch.tensor(y, dtype=torch.long),
                'personalized_feat': p
            }
    
    
        
        
