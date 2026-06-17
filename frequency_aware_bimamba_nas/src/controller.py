import os
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_NAMES = [
    'conv1x1', 'conv1x3', 'skip', 'none',
    'maxpool3', 'avgpool3', 'depthwise_conv3x1', 'se_block'
]

NON_NONE_OP_NAMES = [op for op in OP_NAMES if op != 'none']
NON_NONE_OP_INDICES = [OP_NAMES.index(op) for op in NON_NONE_OP_NAMES]


class RLController(nn.Module):
    def __init__(self, max_nodes=6):
        super().__init__()
        self.max_nodes = max_nodes

        self.d_state_options = [16, 32, 64, 96]
        self.d_conv_options = [4, 8, 16]
        self.expand_options = [2, 4, 6]
        self.fusion_mode_options = ['projection', 'add', 'weighted', 'concat']

        self.d_state_emb = nn.Embedding(len(self.d_state_options), 32)
        self.d_conv_emb = nn.Embedding(len(self.d_conv_options), 32)
        self.expand_emb = nn.Embedding(len(self.expand_options), 32)
        self.fusion_mode_emb = nn.Embedding(len(self.fusion_mode_options), 32)

        self.lstm = nn.LSTMCell(128, 100)

        self.num_nodes_layer = nn.Linear(100, max_nodes - 2)
        self.edge_op_layer = nn.Linear(100, len(OP_NAMES))
        self.d_state_layer = nn.Linear(100, len(self.d_state_options))
        self.d_conv_layer = nn.Linear(100, len(self.d_conv_options))
        self.expand_layer = nn.Linear(100, len(self.expand_options))
        self.fusion_mode_layer = nn.Linear(100, len(self.fusion_mode_options))

        self.saved_log_probs = None
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.h = None
        self.c = None
        self.saved_log_probs = None

    def _sample_from_logits(self, logits):
        probs = F.softmax(logits, dim=-1)
        idx = probs.multinomial(1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, idx)
        return idx, log_prob

    def _sample_non_none_op(self, logits):
        masked_logits = logits[:, NON_NONE_OP_INDICES]
        local_idx, log_prob = self._sample_from_logits(masked_logits)
        op_idx = NON_NONE_OP_INDICES[local_idx.item()]
        return OP_NAMES[op_idx], log_prob

    def sample_architecture(self, device=None, subject_id=None):
        if device is None:
            device = next(self.parameters()).device
        else:
            self.to(device)

        if self.h is None:
            self.h = torch.zeros(1, 100, device=device)
            self.c = torch.zeros(1, 100, device=device)

        log_probs = []

        logits = self.num_nodes_layer(self.h)
        num_nodes_idx, num_nodes_log_prob = self._sample_from_logits(logits)
        log_probs.append(num_nodes_log_prob)
        num_nodes = 3 + num_nodes_idx.item()

        edge_ops_matrix = [['none' for _ in range(num_nodes)] for _ in range(num_nodes)]
        node_configs = []

        for i in range(num_nodes):
            node_features = []

            logits = self.d_state_layer(self.h)
            d_state_idx, d_state_log_prob = self._sample_from_logits(logits)
            log_probs.append(d_state_log_prob)
            d_state = self.d_state_options[d_state_idx.item()]
            node_features.append(self.d_state_emb(d_state_idx.squeeze()))

            logits = self.d_conv_layer(self.h)
            d_conv_idx, d_conv_log_prob = self._sample_from_logits(logits)
            log_probs.append(d_conv_log_prob)
            d_conv = self.d_conv_options[d_conv_idx.item()]
            node_features.append(self.d_conv_emb(d_conv_idx.squeeze()))

            logits = self.expand_layer(self.h)
            expand_idx, expand_log_prob = self._sample_from_logits(logits)
            log_probs.append(expand_log_prob)
            expand = self.expand_options[expand_idx.item()]
            node_features.append(self.expand_emb(expand_idx.squeeze()))

            logits = self.fusion_mode_layer(self.h)
            fusion_mode_idx, fusion_mode_log_prob = self._sample_from_logits(logits)
            log_probs.append(fusion_mode_log_prob)
            fusion_mode = self.fusion_mode_options[fusion_mode_idx.item()]
            node_features.append(self.fusion_mode_emb(fusion_mode_idx.squeeze()))

            node_feature = torch.cat(node_features, dim=-1)
            self.h, self.c = self.lstm(node_feature.unsqueeze(0), (self.h, self.c))

            node_configs.append({
                'd_state': d_state,
                'd_conv': d_conv,
                'expand': expand,
                'fusion_mode': fusion_mode
            })

            if i > 0:
                for j in range(i):
                    logits = self.edge_op_layer(self.h)
                    must_connect = j == i - 1 and all(
                        edge_ops_matrix[k][i] == 'none' for k in range(i - 1)
                    )

                    if must_connect:
                        op_name, op_log_prob = self._sample_non_none_op(logits)
                    else:
                        op_idx, op_log_prob = self._sample_from_logits(logits)
                        op_name = OP_NAMES[op_idx.item()]

                    log_probs.append(op_log_prob)
                    edge_ops_matrix[j][i] = op_name

        self.saved_log_probs = torch.cat(log_probs)

        if subject_id is not None:
            print(f'Sampled edge_ops_matrix for subject {subject_id}:')
            for row in edge_ops_matrix:
                print(row)
            print(f'Sampled node_configs for subject {subject_id}:')
            for node_config in node_configs:
                print(node_config)

        return {
            'num_nodes': num_nodes,
            'edge_ops_matrix': edge_ops_matrix,
            'node_configs': node_configs
        }

    def get_log_prob(self):
        if self.saved_log_probs is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return self.saved_log_probs.sum()
