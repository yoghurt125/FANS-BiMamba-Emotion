import os
import math
import numpy as np
import scipy.io
import torch
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset, DataLoader, Sampler


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_de(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance + 1e-8) / 2

#DEAP process
class DEAP_DE_Preprocessor:
    def __init__(self, raw_data_path, processed_dir, label_type='valence'):
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir
        self.num_subjects = 32
        self.fs = 128
        self.bands = {
            'theta': (4, 8),
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 45)
        }
        label_map = {'valence': 0, 'arousal': 1}
        if label_type not in label_map:
            raise ValueError(f"label_type must be one of {sorted(label_map)}, got {label_type}")
        self.label_type = label_type
        self.label_index = label_map[label_type]
        os.makedirs(self.processed_dir, exist_ok=True)
        for band in self.bands.keys():
            os.makedirs(os.path.join(self.processed_dir, band), exist_ok=True)
        print(f"DEAP DE Preprocessor initialized.")
        print(f"Raw data path: {self.raw_data_path}")
        print(f"Processed data will be saved to: {self.processed_dir}")

    def process_single_subject(self, subject_id):
        s_id_str = f's{subject_id:02d}'
        print(f"Processing subject {s_id_str}...")
        
        input_file = os.path.join(self.raw_data_path, f'{s_id_str}.mat')
        raw_data = scipy.io.loadmat(input_file)
        data = raw_data['data']   # Shape: (40, 32, 8064)
        labels = raw_data['labels'] # Shape: (40, 4)

        # Process each frequency band separately
        for band_name, band_range in self.bands.items():
            output_file = os.path.join(self.processed_dir, band_name, f'{s_id_str}.mat')
            if os.path.exists(output_file):
                print(f"Subject {s_id_str} band {band_name} has already been processed. Skipping.")
                continue

            subject_x = []
            subject_y = []

            for trial_idx in range(data.shape[0]):
                trial_data = data[trial_idx] # Shape: (32, 8064)
                
                # Baseline DE for this band
                base_signal = trial_data[:, :384] # 3s baseline
                base_de = np.zeros(32) # channels
                for ch_idx in range(32):
                    filtered = butter_bandpass_filter(base_signal[ch_idx, :], band_range[0], band_range[1], self.fs, order=3)
                    base_de[ch_idx] = compute_de(filtered)
                
                # Main signal DE for this band
                main_signal = trial_data[:, 384:] # 60s signal
                num_windows = main_signal.shape[1] // self.fs
                trial_de_sequence = np.zeros((num_windows, 32)) # (60, 32)
                
                for win_idx in range(num_windows):
                    window_de = np.zeros(32) # channels
                    window_signal = main_signal[:, win_idx*self.fs : (win_idx+1)*self.fs]
                    for ch_idx in range(32):
                        filtered = butter_bandpass_filter(window_signal[ch_idx, :], band_range[0], band_range[1], self.fs, order=3)
                        window_de[ch_idx] = compute_de(filtered)
                    trial_de_sequence[win_idx, :] = window_de

                # Subtract baseline
                trial_features = trial_de_sequence - base_de # Broadcasting
                
                # Binary label: 1 if the selected DEAP score is greater than 5.
                label = 1 if labels[trial_idx, self.label_index] > 5 else 0
                
                subject_x.append(trial_features.T) # Shape: (32, 60)
                subject_y.append(label)

            final_x = np.stack(subject_x) # Shape: (40, 32, 60)
            final_y = np.array(subject_y, dtype=np.int64) # Shape: (40,)

            scipy.io.savemat(output_file, {'data': final_x, 'labels': final_y})
            print(f"Saved processed {band_name} data for subject {s_id_str} to {output_file}")

    def process_all_subjects(self):
        print("\nStarting DE feature extraction for all subjects...")
        for subject_id in range(1, self.num_subjects + 1):
            self.process_single_subject(subject_id)
        print("All subjects processed and saved.")


class DEAP_DE_Dataset(Dataset):
    def __init__(self, processed_dir, subject_ids, is_train=False):
        self.X_list = []
        self.y_list = []
        self.is_train = is_train
        for subject_id in subject_ids:
            file_path = os.path.join(processed_dir, f's{subject_id:02d}.mat')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Processed file not found for subject {subject_id}: {file_path}")
            data = scipy.io.loadmat(file_path)
            # data['data'] shape: (40, 128, 60), data['labels'] shape: (40, 1) or (40,)
            self.X_list.append(data['data'])
            self.y_list.append(data['labels'].flatten()) 
            
        self.X = torch.FloatTensor(np.concatenate(self.X_list, axis=0))
        self.y = torch.LongTensor(np.concatenate(self.y_list, axis=0))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.is_train:
            scale = torch.empty(1).uniform_(0.95, 1.05).item()
            x = x * scale
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        return x, y

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, fill_last_batch=True):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.fill_last_batch = fill_last_batch
        self.class_indices = {
            int(cls): np.where(self.labels == cls)[0].tolist()
            for cls in np.unique(self.labels)
        }
        if len(self.class_indices) == 0:
            raise ValueError('labels must not be empty')
        self.num_samples = len(self.labels)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        counts = {cls: len(idxs) for cls, idxs in self.class_indices.items()}
        total = sum(counts.values())
        self.batch_class_counts = {
            cls: max(1, int(round(count / total * self.batch_size)))
            for cls, count in counts.items()
        }
        diff = self.batch_size - sum(self.batch_class_counts.values())
        if diff != 0:
            majority_cls = max(counts, key=counts.get)
            self.batch_class_counts[majority_cls] += diff
        self.batch_class_counts = {
            cls: max(0, n) for cls, n in self.batch_class_counts.items()
        }
        self.reset()

    def reset(self):
        self.shuffled_class_indices = {}
        for cls, idxs in self.class_indices.items():
            idxs = np.asarray(idxs)
            np.random.shuffle(idxs)
            self.shuffled_class_indices[cls] = idxs.tolist()
        self.class_pointers = {cls: 0 for cls in self.class_indices}

    def __iter__(self):
        self.reset()
        finished = False
        while not finished:
            batch = []
            for cls, n in self.batch_class_counts.items():
                if n <= 0:
                    continue
                start = self.class_pointers[cls]
                end = min(start + n, len(self.shuffled_class_indices[cls]))
                batch.extend(self.shuffled_class_indices[cls][start:end])
                self.class_pointers[cls] = end
            if len(batch) == 0:
                break
            if len(batch) < self.batch_size and self.fill_last_batch:
                needed = self.batch_size - len(batch)
                for cls in self.class_indices:
                    remain = len(self.shuffled_class_indices[cls]) - self.class_pointers[cls]
                    take = min(needed, remain)
                    if take > 0:
                        start = self.class_pointers[cls]
                        batch.extend(self.shuffled_class_indices[cls][start:start + take])
                        self.class_pointers[cls] += take
                        needed -= take
                    if needed <= 0:
                        break
            yield batch[:self.batch_size] if self.fill_last_batch else batch
            finished = all(
                self.class_pointers[cls] >= len(self.shuffled_class_indices[cls])
                for cls in self.class_indices
            )

    def __len__(self):
        return self.num_batches

def prepare_data_loso(processed_dir, test_subject, batch_size=32):
    """
    Prepare leave-one-subject-out data loaders with stratified sampling for both training and test sets.
    """
    all_subjects = list(range(1, 33))
    train_subjects = [s for s in all_subjects if s != test_subject]
    test_subjects = [test_subject]

    print(f"\nPreparing LOSO data loaders: Test Subject = {test_subject}")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print(f"Data directory: {processed_dir}")

    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    test_file_path = os.path.join(processed_dir, f's{test_subject:02d}.mat')
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test subject file not found: {test_file_path}")

    train_dataset = DEAP_DE_Dataset(processed_dir, train_subjects, is_train=True)
    test_dataset = DEAP_DE_Dataset(processed_dir, test_subjects, is_train=False)

    
    train_labels = train_dataset.y.numpy()
    test_labels = test_dataset.y.numpy()
    
    # Use stratified sampling for training and testing.
    train_sampler = StratifiedBatchSampler(train_labels, batch_size, fill_last_batch=True)
    test_sampler = StratifiedBatchSampler(test_labels, batch_size, fill_last_batch=False)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    return train_loader, test_loader

def prepare_multiband_loso(processed_root_dir, test_subject, bands=None, batch_size=32):
    if bands is None:
        bands = ['theta', 'alpha', 'beta', 'gamma']
    all_subjects = list(range(1, 33))
    train_subjects = [s for s in all_subjects if s != test_subject]
    test_subjects = [test_subject]

    train_dataset = MultiBandDEAPDataset(processed_root_dir, train_subjects, bands=bands, is_train=True)
    test_dataset = MultiBandDEAPDataset(processed_root_dir, test_subjects, bands=bands, is_train=False)

    train_labels = train_dataset.y.numpy()
    test_labels = test_dataset.y.numpy()

    train_sampler = StratifiedBatchSampler(train_labels, batch_size, fill_last_batch=True)
    test_sampler = StratifiedBatchSampler(test_labels, batch_size, fill_last_batch=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    return train_loader, test_loader, train_labels, test_labels

class MultiBandDEAPDataset(Dataset):
    def __init__(self, processed_dir, subject_ids, bands=['theta', 'alpha', 'beta', 'gamma'], is_train=False):
        self.X_list = []
        self.y_list = []
        self.is_train = is_train
        for subject_id in subject_ids:
            band_data = []
            reference_labels = None
            for band in bands:
                file_path = os.path.join(processed_dir, band, f's{subject_id:02d}.mat')
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Processed file not found for subject {subject_id} band {band}: {file_path}")
                data = scipy.io.loadmat(file_path)
                labels = data['labels'].flatten()
                if reference_labels is None:
                    reference_labels = labels
                elif not np.array_equal(reference_labels, labels):
                    raise ValueError(f"Label mismatch across bands for subject {subject_id}. Check preprocessing output.")
                # data['data'] shape: (40, 32, 60)
                band_data.append(data['data'])

            band_data = np.concatenate(band_data, axis=1)
            self.X_list.append(band_data)
            self.y_list.append(reference_labels)
        self.X = torch.FloatTensor(np.concatenate(self.X_list, axis=0))
        self.y = torch.LongTensor(np.concatenate(self.y_list, axis=0))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.is_train:
            # Data augmentation
            scale = torch.empty(1).uniform_(0.95, 1.05).item()
            x = x * scale
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        return x, y

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess DEAP EEG into band-wise DE features.')
    parser.add_argument('--raw_data_path', type=str, required=True, help='Directory containing DEAP s01.mat ... s32.mat files.')
    parser.add_argument('--processed_dir', type=str, required=True, help='Output directory for processed band-wise .mat files.')
    parser.add_argument('--label_type', type=str, default='valence', choices=['valence', 'arousal'], help='DEAP label dimension to binarize.')
    parser.add_argument('--test_loader', action='store_true', help='Run a small loader sanity check after preprocessing.')
    args = parser.parse_args()

    preprocessor = DEAP_DE_Preprocessor(args.raw_data_path, args.processed_dir, label_type=args.label_type)
    preprocessor.process_all_subjects()

    if args.test_loader:
        print('\nTesting LOSO data loader for subject 1...')
        train_loader, test_loader = prepare_data_loso(args.processed_dir, test_subject=1, batch_size=10)
        x_train, y_train = next(iter(train_loader))
        print(f'Train batch shape: X={x_train.shape}, y={y_train.shape}')
        x_test, y_test = next(iter(test_loader))
        print(f'Test batch shape: X={x_test.shape}, y={y_test.shape}')
        print('Data preparation script finished successfully.')
