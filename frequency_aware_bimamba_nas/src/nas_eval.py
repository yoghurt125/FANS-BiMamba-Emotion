import os
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'

import copy
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dag_search_space import DAGSearchSpace, MultiHeadFrequencyFusion, GatedFrequencyFusion, SelfAttentionPooling
from deap_dataset import MultiBandDEAPDataset, StratifiedBatchSampler
from utils import FocalLoss


CHECKPOINT_VERSION = 3


def setup_logger(log_dir, name='nas_eval'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(name)


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, band_architectures, input_dim=32, num_heads=8, num_classes=2, fusion_mode='multihead'):
        super().__init__()
        self.band_models = nn.ModuleDict()
        for band, arch in band_architectures.items():
            self.band_models[band] = DAGSearchSpace(
                input_dim=input_dim,
                num_nodes=arch['num_nodes'],
                node_configs=arch['node_configs'],
                edge_ops_matrix=arch['edge_ops_matrix'],
                freq_band=band,
                use_classifier=False
            )

        assert fusion_mode in ['multihead', 'gated'], f'Unsupported fusion_mode: {fusion_mode}'
        self.fusion_mode = fusion_mode

        if fusion_mode == 'multihead':
            self.fusion = MultiHeadFrequencyFusion(dim=input_dim, num_bands=4, num_heads=num_heads, dropout=0.1)
            self.pool = SelfAttentionPooling(input_dim)
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            self.fusion = GatedFrequencyFusion(dim=input_dim, num_bands=4, dropout=0.1)
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, band_inputs):
        band_feats = {band: model(band_inputs[band]) for band, model in self.band_models.items()}
        if self.fusion_mode == 'multihead':
            fused_seq = self.fusion(band_feats)
            pooled = self.pool(fused_seq)
            return self.classifier(pooled)
        pooled = self.fusion(band_feats)
        return self.classifier(pooled)


def split_train_val_subjects(test_subject, val_fraction=0.2, seed=42):
    all_subjects = list(range(1, 33))
    candidate_subjects = [subject for subject in all_subjects if subject != test_subject]
    rng = np.random.default_rng(seed + test_subject)
    shuffled = np.array(candidate_subjects)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_subjects = sorted(shuffled[:val_count].tolist())
    train_subjects = sorted(shuffled[val_count:].tolist())
    return train_subjects, val_subjects, [test_subject]


def make_multiband_loader(processed_root_dir, subject_ids, bands, batch_size, is_train, fill_last_batch):
    dataset = MultiBandDEAPDataset(processed_root_dir, subject_ids, bands=bands, is_train=is_train)
    labels = dataset.y.numpy()

    if is_train:
        sampler = StratifiedBatchSampler(labels, batch_size, fill_last_batch=fill_last_batch)
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader, labels


def build_band_inputs(inputs, bands, device):
    band_inputs = {}
    for idx, band in enumerate(bands):
        start = idx * 32
        end = (idx + 1) * 32
        band_inputs[band] = inputs[:, start:end, :].to(device)
    return band_inputs


def train_one_epoch(model, loader, criterion, optimizer, bands, device):
    model.train()
    criterion.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        labels = labels.to(device)
        band_inputs = build_band_inputs(inputs, bands, device)

        optimizer.zero_grad()
        outputs = model(band_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(labels).sum().item()

    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate_model(model, loader, criterion, bands, device):
    model.eval()
    criterion.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            labels = labels.to(device)
            band_inputs = build_band_inputs(inputs, bands, device)
            outputs = model(band_inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return loss_sum / max(total, 1), acc, f1, cm, all_labels, all_preds


def get_recalls_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall_0, recall_1


def load_checkpoint(results_dir):
    checkpoint_path = os.path.join(results_dir, 'loso_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
        return None
    return checkpoint


def save_checkpoint(results_dir, completed_subjects, fold_results):
    checkpoint_path = os.path.join(results_dir, 'loso_checkpoint.pth')
    checkpoint = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'completed_subjects': completed_subjects,
        'fold_results': fold_results
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved. Completed subjects: {len(completed_subjects)}')


def main():
    import random

    parser = argparse.ArgumentParser(description='Evaluate searched band architectures under LOSO.')
    parser.add_argument('--processed_de_dir', type=str, required=True, help='Root processed DEAP directory containing theta/alpha/beta/gamma folders.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing best_arch_{band}.pth and output logs/results.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--fusion_mode', type=str, default='multihead', choices=['multihead', 'gated'])
    parser.add_argument('--selection_metric', type=str, default='val_acc', choices=['val_acc', 'val_f1'])
    parser.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta', 'gamma'])
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    processed_de_dir = args.processed_de_dir
    results_dir = args.results_dir
    bands = args.bands
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    patience = args.patience
    selection_metric = args.selection_metric

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_logger = setup_logger(results_dir, name='nas_eval')
    eval_logger.info(f'Using device: {device}')

    best_architectures = {}
    architecture_metadata = {}
    for band in bands:
        arch_path = os.path.join(results_dir, f'best_arch_{band}.pth')
        assert os.path.exists(arch_path), f'Best architecture file not found: {arch_path}'
        arch_data = torch.load(arch_path, map_location='cpu')
        best_architectures[band] = arch_data['arch_config']
        architecture_metadata[band] = {
            key: value for key, value in arch_data.items() if key != 'arch_config'
        }

    checkpoint = load_checkpoint(results_dir)
    if checkpoint is not None:
        completed_subjects = checkpoint['completed_subjects']
        fold_results = checkpoint['fold_results']
        eval_logger.info(f'Resuming evaluation. Completed subjects: {completed_subjects}')
    else:
        completed_subjects = []
        fold_results = []
        eval_logger.info('Starting a new LOSO evaluation.')

    for test_subject in range(1, 33):
        if test_subject in completed_subjects:
            eval_logger.info(f'Skipping completed subject {test_subject}')
            continue

        train_subjects, val_subjects, test_subjects = split_train_val_subjects(
            test_subject=test_subject,
            val_fraction=0.2,
            seed=seed
        )

        train_loader, train_labels = make_multiband_loader(
            processed_de_dir, train_subjects, bands, batch_size, is_train=True, fill_last_batch=True
        )
        val_loader, val_labels = make_multiband_loader(
            processed_de_dir, val_subjects, bands, batch_size, is_train=False, fill_last_batch=False
        )
        test_loader, test_labels = make_multiband_loader(
            processed_de_dir, test_subjects, bands, batch_size, is_train=False, fill_last_batch=False
        )

        eval_logger.info(f'LOSO fold {test_subject[0] if isinstance(test_subject, list) else test_subject}')
        eval_logger.info(f'Train subjects: {train_subjects}')
        eval_logger.info(f'Validation subjects: {val_subjects}')
        eval_logger.info(f'Test subject: {test_subjects}')
        eval_logger.info(f'Train label distribution: {np.bincount(train_labels, minlength=2)}')
        eval_logger.info(f'Validation label distribution: {np.bincount(val_labels, minlength=2)}')
        eval_logger.info(f'Test label distribution: {np.bincount(test_labels, minlength=2)}')

        model = CrossAttentionFusionModel(
            best_architectures,
            input_dim=32,
            num_heads=args.num_heads,
            num_classes=2,
            fusion_mode=args.fusion_mode
        ).to(device)

        criterion = FocalLoss(alpha=0.45, gamma=3.0).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=2e-4)

        best_state = None
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, bands, device)
            val_loss, val_acc, val_f1, val_cm, _, _ = evaluate_model(model, val_loader, criterion, bands, device)

            eval_logger.info(
                f'Epoch {epoch + 1}/{max_epochs} | '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}% | Val F1: {val_f1:.4f}'
            )

            improved = val_acc > best_val_acc if selection_metric == 'val_acc' else val_f1 > best_val_f1
            if improved:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                eval_logger.info(
                    f'New best validation checkpoint at epoch {best_epoch}. '
                    f'Val Acc: {best_val_acc * 100:.2f}%, Val F1: {best_val_f1:.4f}'
                )
            else:
                patience_counter += 1
                eval_logger.info(f'No validation improvement for {patience_counter} epochs.')

            if patience_counter >= patience:
                eval_logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_acc, test_f1, test_cm, test_labels_epoch, test_preds_epoch = evaluate_model(
            model, test_loader, criterion, bands, device
        )
        recall_0, recall_1 = get_recalls_from_cm(test_cm)
        pred_dist = test_cm.sum(axis=0)
        subject_id = int(test_subjects[0])

        fold_result = {
            'subject': subject_id,
            'best_epoch': int(best_epoch),
            'best_val_accuracy': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'test_loss': float(test_loss),
            'test_recall_0': float(recall_0),
            'test_recall_1': float(recall_1),
            'test_confusion_matrix': test_cm.tolist(),
            'test_prediction_distribution': pred_dist.tolist(),
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects
        }
        fold_results.append(fold_result)
        completed_subjects.append(subject_id)

        eval_logger.info(
            f'LOSO fold {subject_id} final test results | '
            f'Best Epoch: {best_epoch} | Test Acc: {test_acc * 100:.2f}% | '
            f'Test F1: {test_f1:.4f} | Recall_1: {recall_1:.4f} | Recall_0: {recall_0:.4f}'
        )
        eval_logger.info(f'Test confusion matrix:\n{test_cm}')
        eval_logger.info(f'Subject {subject_id}: prediction distribution: {pred_dist}')

        save_checkpoint(results_dir, completed_subjects, fold_results)
        eval_logger.info(f'Progress: {len(completed_subjects)}/32 subjects completed')
        eval_logger.info('=' * 80)

    all_accuracies = [item['test_accuracy'] for item in fold_results]
    all_f1s = [item['test_f1'] for item in fold_results]
    all_recall_1 = [item['test_recall_1'] for item in fold_results]
    all_recall_0 = [item['test_recall_0'] for item in fold_results]

    final_results = {
        'fold_results': fold_results,
        'architecture_metadata': architecture_metadata,
        'mean_accuracy': float(np.mean(all_accuracies)),
        'std_accuracy': float(np.std(all_accuracies)),
        'mean_f1': float(np.mean(all_f1s)),
        'std_f1': float(np.std(all_f1s)),
        'mean_recall_1': float(np.mean(all_recall_1)),
        'std_recall_1': float(np.std(all_recall_1)),
        'mean_recall_0': float(np.mean(all_recall_0)),
        'std_recall_0': float(np.std(all_recall_0))
    }

    print('\nFinal LOSO Results:')
    print(f"Mean Accuracy: {final_results['mean_accuracy'] * 100:.2f}%")
    print(f"Std Accuracy: {final_results['std_accuracy'] * 100:.2f}%")
    print(f"Mean F1: {final_results['mean_f1']:.4f}")
    print(f"Std F1: {final_results['std_f1']:.4f}")
    print(f"Mean Recall 1: {final_results['mean_recall_1']:.4f}")
    print(f"Std Recall 1: {final_results['std_recall_1']:.4f}")
    print(f"Mean Recall 0: {final_results['mean_recall_0']:.4f}")
    print(f"Std Recall 0: {final_results['std_recall_0']:.4f}")

    torch.save(final_results, os.path.join(results_dir, 'final_results_crossattn_eval.pth'))
    with open(os.path.join(results_dir, 'final_results_crossattn_eval.json'), 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=True)

    checkpoint_path = os.path.join(results_dir, 'loso_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        eval_logger.info('Evaluation completed. Checkpoint file removed.')


if __name__ == '__main__':
    main()
