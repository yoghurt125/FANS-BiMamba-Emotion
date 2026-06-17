import os
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import argparse
from datetime import datetime
from deap_dataset import DEAP_DE_Dataset
from torch.utils.data import DataLoader
from controller import RLController
from dag_search_space import DAGSearchSpace
from sklearn.metrics import f1_score
from utils import FocalLoss


def setup_logger(log_dir, name='nas_search'):
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


def train_model(model, train_loader, val_loader, device, logger, epochs=80, model_desc='Model', early_stopping_patience=15):
    model = model.to(device)
    criterion = FocalLoss(alpha=0.45, gamma=3.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if targets.ndim == 2:
                targets = targets.argmax(dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = targets.size(0)
            train_loss_sum += loss.item() * batch_size
            _, predicted = outputs.max(1)
            train_total += batch_size
            train_correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if targets.ndim == 2:
                    targets = targets.argmax(dim=1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_size = targets.size(0)
                val_loss_sum += loss.item() * batch_size
                _, predicted = outputs.max(1)
                val_total += batch_size
                val_correct += predicted.eq(targets).sum().item()
                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0

        logger.info(
            f'Epoch {epoch + 1}/{epochs} | {model_desc} | '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}% | Val F1: {val_f1:.4f}'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
            logger.info(f'New best validation accuracy: {best_val_acc * 100:.2f}%')
        else:
            patience_counter += 1
            logger.info(f'No validation improvement for {patience_counter} epochs.')

        if patience_counter >= early_stopping_patience:
            logger.info(f'Early stopping triggered at epoch {epoch + 1}')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_acc


def search_best_architecture(train_loader, val_loader, device, logger, save_dir, freq_band, search_train_subjects=None, search_val_subjects=None, num_steps=50, train_epochs_search=80, search_patience=15):
    controller = RLController().to(device)
    controller_optimizer = optim.Adam(controller.parameters(), lr=0.001)
    baseline = 0.0
    best_val_acc = 0.0
    best_arch = None
    results_file = os.path.join(save_dir, f'nas_search_results_{freq_band}.txt')

    logger.info(f"\n{'=' * 20} Starting architecture search for {freq_band} band {'=' * 20}")

    for step in range(num_steps):
        controller.reset_hidden_state()
        start_time = time.time()
        arch_config = controller.sample_architecture(device)
        logger.info(f"\n--- Step {step + 1}/{num_steps} | Sampling architecture for {freq_band}: {arch_config} ---")

        model = DAGSearchSpace(
            input_dim=32,
            num_nodes=arch_config['num_nodes'],
            node_configs=arch_config['node_configs'],
            edge_ops_matrix=arch_config['edge_ops_matrix'],
            freq_band=freq_band
        )

        val_acc = train_model(
            model,
            train_loader,
            val_loader,
            device,
            logger,
            epochs=train_epochs_search,
            model_desc=f'Arch-{step + 1}-{freq_band}',
            early_stopping_patience=search_patience
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_arch = arch_config
            torch.save({
                'arch_config': best_arch,
                'validation_accuracy': best_val_acc,
                'freq_band': freq_band,
                'search_protocol': 'fixed_subject_level_train_validation_split',
                'search_train_subjects': search_train_subjects,
                'search_val_subjects': search_val_subjects
            }, os.path.join(save_dir, f'best_arch_{freq_band}.pth'))
            logger.info(f'New best architecture for {freq_band}. Val Acc: {best_val_acc * 100:.2f}%')

        reward = float(val_acc - baseline)
        baseline = 0.9 * baseline + 0.1 * val_acc

        controller_optimizer.zero_grad()
        log_prob = controller.get_log_prob()
        loss = -reward * log_prob
        loss.backward()
        controller_optimizer.step()

        elapsed = time.time() - start_time
        logger.info(
            f'Step {step + 1} took {elapsed:.2f}s. '
            f'Val Acc: {val_acc * 100:.2f}%. Best so far: {best_val_acc * 100:.2f}%'
        )

        with open(results_file, 'a') as f:
            f.write(f"Step {step + 1}: ValAcc={val_acc:.4f}, Arch={arch_config}\n")

    logger.info(f"{'=' * 20} Architecture search complete for {freq_band} {'=' * 20}")
    return best_arch


def main():
    import random

    parser = argparse.ArgumentParser(description='Run single-band NAS for DEAP band-wise DE features.')
    parser.add_argument('--processed_de_dir', type=str, required=True, help='Root processed DEAP directory containing theta/alpha/beta/gamma folders.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory for searched architectures and logs.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--search_epochs', type=int, default=80)
    parser.add_argument('--search_patience', type=int, default=15)
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
    logger = setup_logger(results_dir, name='nas_search_singleband')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    frequency_bands = args.bands

    for band in frequency_bands:
        logger.info(f"\n==== Searching best architecture for {band} ====")
        all_subjects = list(range(1, 33))
        rng = np.random.default_rng(seed)
        rng.shuffle(all_subjects)
        train_subjects = sorted(all_subjects[:16])
        val_subjects = sorted(all_subjects[16:])

        logger.info(f'Train subjects: {train_subjects}')
        logger.info(f'Validation subjects: {val_subjects}')

        train_set = DEAP_DE_Dataset(os.path.join(processed_de_dir, band), train_subjects, is_train=True)
        val_set = DEAP_DE_Dataset(os.path.join(processed_de_dir, band), val_subjects, is_train=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        best_arch = search_best_architecture(
            train_loader,
            val_loader,
            device,
            logger,
            results_dir,
            band,
            train_subjects,
            val_subjects,
            num_steps=args.num_steps,
            train_epochs_search=args.search_epochs,
            search_patience=args.search_patience,
        )
        logger.info(f'Best architecture for {band}: {best_arch}')

    logger.info('All single-band architecture search finished. You can now run nas_eval.py.')


if __name__ == '__main__':
    main()
