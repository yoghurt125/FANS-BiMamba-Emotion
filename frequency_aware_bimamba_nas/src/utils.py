import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.45, gamma=3.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.prediction_history = []

        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha_tensor', alpha_tensor)
            self.alpha = None
        else:
            self.register_buffer('alpha_tensor', torch.empty(0), persistent=False)
            self.alpha = float(alpha)

    def forward(self, inputs, targets):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        targets = targets.long().view(-1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha_tensor.numel() > 0:
            alpha = self.alpha_tensor.to(device=inputs.device, dtype=inputs.dtype)
            alpha_t = alpha.gather(0, targets)
        elif inputs.size(-1) == 2:
            alpha_t = torch.where(
                targets == 1,
                torch.full_like(ce_loss, self.alpha),
                torch.full_like(ce_loss, 1.0 - self.alpha),
            )
        else:
            alpha_t = torch.full_like(ce_loss, self.alpha)

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce_loss

        if self.training:
            with torch.no_grad():
                predictions = torch.argmax(inputs, dim=1)
                pred_dist = torch.bincount(predictions, minlength=inputs.size(-1))
                self.prediction_history.append(pred_dist.cpu().numpy())

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

    def get_prediction_bias(self):
        if not self.prediction_history:
            return None

        recent_preds = np.array(self.prediction_history[-10:])
        if recent_preds.ndim != 2 or recent_preds.shape[1] < 2:
            return None

        totals = recent_preds[:, 0] + recent_preds[:, 1]
        totals = np.maximum(totals, 1)
        class_0_ratio = recent_preds[:, 0] / totals

        return {
            'mean_class_0_ratio': float(np.mean(class_0_ratio)),
            'std_class_0_ratio': float(np.std(class_0_ratio)),
            'bias_detected': bool(np.std(class_0_ratio) < 0.1),
        }


def save_architecture(arch_config, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(arch_config, f, ensure_ascii=True, indent=2)


def load_architecture(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
