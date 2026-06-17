import os
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from spatial_feature_extractor import SpatialFeatureExtractor
from bidirectional_mamba import BidirectionalMamba

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, l = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


OPS = {
    'conv1x1': lambda in_dim, out_dim: nn.Conv1d(in_dim, out_dim, kernel_size=1, padding=0),
    'conv1x3': lambda in_dim, out_dim: nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
    'skip': lambda in_dim, out_dim: nn.Identity() if in_dim == out_dim else nn.Conv1d(in_dim, out_dim, 1),
    'none': lambda in_dim, out_dim: None,
    'maxpool3': lambda in_dim, out_dim: nn.Sequential(
        nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity(),
        nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
    ),
    'avgpool3': lambda in_dim, out_dim: nn.Sequential(
        nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity(),
        nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
    ),
    'depthwise_conv3x1': lambda in_dim, out_dim: nn.Sequential(
        nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
        nn.Conv1d(in_dim, out_dim, 1)
    ),
    'se_block': lambda in_dim, out_dim: nn.Sequential(
        nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity(),
        SEBlock(out_dim)
    )
}

class DAGNode(nn.Module):
    def __init__(self, in_dim, out_dim, mamba_config, dropout_p=0.5):
        super().__init__()
        # Use bidirectional Mamba instead of single direction
        fusion_mode = mamba_config.get('fusion_mode', 'projection')  # Default to projection
        self.mamba = BidirectionalMamba(
            d_model=in_dim,
            d_state=mamba_config['d_state'],
            d_conv=mamba_config['d_conv'],
            expand=mamba_config['expand'],
            fusion_mode=fusion_mode
        )
        # Adjust normalization dimension based on fusion mode
        norm_dim = self.mamba.output_dim
        self.norm = nn.LayerNorm(norm_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        
        # Add projection if output dimension changed
        if norm_dim != out_dim:
            self.proj = nn.Linear(norm_dim, out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        x = self.mamba(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        attn_weights = F.softmax(self.attention(x), dim=1)
        # attn_weights shape: (batch, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)
        # pooled shape: (batch, input_dim)
        return pooled

class MultiHeadFrequencyFusion(nn.Module):
    """
    Multi-Query Frequency Fusion with band-specific attention
    """
    def __init__(self, dim, num_bands=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        # Band-specific query projections
        self.band_q_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_bands)
        ])
        
        # Shared key and value projections
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Band-specific output projections
        self.band_out_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_bands)
        ])
        
        # Multi-query fusion network
        self.query_fusion = nn.Sequential(
            nn.Linear(dim * num_bands, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # Gate network for band importance
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_bands, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, num_bands),
            nn.Sigmoid()
        )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_bands) / num_bands)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        self.scale = self.head_dim ** -0.5
        
    def band_specific_attention(self, query, key_value_stack, band_idx):
        """Band-specific multi-head attention"""
        B, L, D = query.shape
        
        # Use band-specific query projection
        q = self.band_q_projs[band_idx](query).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Shared key and value projections for all bands
        k = self.k_proj(key_value_stack).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value_stack).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        attn_output = self.band_out_projs[band_idx](attn_output)
        
        return attn_output, attn_weights
    
    def forward(self, band_outputs):
        """
        Args:
            band_outputs: dict, {'theta': tensor, 'alpha': tensor, 'beta': tensor, 'gamma': tensor}
                         tensor shape: [batch, seq_len, dim]
        Returns:
            fused_output: [batch, seq_len, dim]
        """
        bands = ['theta', 'alpha', 'beta', 'gamma']
        band_features = [band_outputs[band] for band in bands]  # List of [batch, seq_len, dim]
        
        B, L, D = band_features[0].shape
        
        # 1. Gate mechanism for band importance
        concatenated = torch.cat(band_features, dim=-1)  # [batch, seq_len, dim*4]
        global_context = torch.mean(concatenated, dim=1)  # [batch, dim*4]
        gate_weights = self.gate_net(global_context)  # [batch, num_bands]
        gate_weights = gate_weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, num_bands, 1]
        
        # 2. Apply gating and learnable fusion weights
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        combined_weights = gate_weights * fusion_weights.view(1, 1, -1, 1)
        
        # Apply weights to band features
        weighted_bands = []
        for i, band_feat in enumerate(band_features):
            weighted = band_feat * combined_weights[:, :, i, :]  # [batch, seq_len, dim]
            weighted_bands.append(weighted)
        
        # 3. Multi-Query Attention: Each band as query
        key_value_stack = torch.stack(weighted_bands, dim=2)  # [batch, seq_len, num_bands, dim]
        key_value_stack = key_value_stack.reshape(B, L * self.num_bands, D)  # [batch, seq_len*num_bands, dim]
        
        # Get attention outputs from each band perspective
        band_attention_outputs = []
        for band_idx, band_query in enumerate(weighted_bands):
            attn_output, _ = self.band_specific_attention(band_query, key_value_stack, band_idx)
            band_attention_outputs.append(attn_output)
        
        # 4. Fuse multi-query attention outputs
        multi_query_concat = torch.cat(band_attention_outputs, dim=-1)  # [batch, seq_len, dim*num_bands]
        
        # Global fusion across sequence dimension
        global_fusion_context = torch.mean(multi_query_concat, dim=1)  # [batch, dim*num_bands]
        fused_output = self.query_fusion(global_fusion_context)  # [batch, dim]
        fused_output = fused_output.unsqueeze(1).expand(B, L, D)  # [batch, seq_len, dim]
        
        # 5. Residual connection with original weighted average
        weighted_avg = torch.mean(torch.stack(weighted_bands, dim=0), dim=0)  # [batch, seq_len, dim]
        output = self.norm1(weighted_avg + self.dropout(fused_output))
        
        # 6. FFN
        ffn_output = self.ffn(output)
        output = self.norm2(output + self.dropout(ffn_output))
        
        return output

class GatedFrequencyFusion(nn.Module):
    """
    Lightweight gated frequency fusion module.
    """
    def __init__(self, dim, num_bands=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_bands, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_bands),
            nn.Sigmoid()
        )
        
        # Feature transforms
        self.band_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_bands)
        ])
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    
    def forward(self, band_outputs):
        bands = ['theta', 'alpha', 'beta', 'gamma']
        band_features = [band_outputs[band] for band in bands]  # List of [batch, seq_len, dim]
        
        B, L, D = band_features[0].shape
        
        # 1. Feature transformation
        transformed_bands = []
        for i, (band_feat, transform) in enumerate(zip(band_features, self.band_transforms)):
            transformed = transform(band_feat)  # [batch, seq_len, dim]
            transformed_bands.append(transformed)
        
        # 2. Gate weight estimation
        concatenated = torch.cat(transformed_bands, dim=-1)  # [batch, seq_len, dim*4]
        global_context = torch.mean(concatenated, dim=1)  # [batch, dim*4]
        gate_weights = self.gate_net(global_context)  # [batch, num_bands]
        
        # 3. Weighted fusion
        weighted_sum = torch.zeros(B, L, D, device=band_features[0].device)
        for i, (band_feat, weight) in enumerate(zip(transformed_bands, gate_weights.unbind(-1))):
            weighted_sum += band_feat * weight.unsqueeze(1).unsqueeze(-1)
        
        # 4. Fusion layer processing
        fused = self.fusion_layer(weighted_sum)  # [batch, seq_len, dim]
        
        # 5. Temporal attention pooling
        attn_weights = F.softmax(self.temporal_attention(fused), dim=1)  # [batch, seq_len, 1]
        pooled = torch.sum(fused * attn_weights, dim=1)  # [batch, dim]
        
        return pooled

class DAGSearchSpace(nn.Module):
    def __init__(self, input_dim, num_nodes, node_configs, edge_ops_matrix, freq_band, num_classes=2, use_classifier=True, dropout_p=0.5):
        super().__init__()
        self.freq_band = freq_band
        self.num_nodes = num_nodes
        self.edge_ops_matrix = edge_ops_matrix
        self.use_classifier = use_classifier
        self.spatial_extractor = SpatialFeatureExtractor(n_channels=input_dim)
        self.nodes = nn.ModuleList([
            DAGNode(input_dim, input_dim, config, dropout_p=dropout_p)
            for config in node_configs
        ])
        self.edge_ops = nn.ModuleDict()
        for i in range(num_nodes):
            for j in range(i):
                op_name = self.edge_ops_matrix[j][i]
                if op_name != 'none':
                    self.edge_ops[f"{j}_{i}"] = OPS[op_name](input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.attention_pooling = SelfAttentionPooling(input_dim)
        #LayerNorm+MLP+Dropout
        self.norm = nn.LayerNorm(input_dim)
        self.head_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(input_dim // 2, num_classes)
        )
        # Redundant classifier removed
        # self.classifier = nn.Linear(input_dim, num_classes)
        
        
        self._init_weights()
    
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  

    def forward(self, x):
        x = self.spatial_extractor(x)
        x = x.transpose(1, 2)
        node_outputs = [None] * self.num_nodes
        for i in range(self.num_nodes):
            inputs = []
            for j in range(i):
                op_name = self.edge_ops_matrix[j][i]
                if op_name != 'none' and node_outputs[j] is not None:
                    key = f"{j}_{i}"
                    if key in self.edge_ops:
                        op = self.edge_ops[key]
                        inp = node_outputs[j].transpose(1, 2)
                        out = op(inp)
                        out = out.transpose(1, 2)
                        inputs.append(out)
            if not inputs:
                node_input = x
            else:
                node_input = torch.stack(inputs).sum(0)
            node_outputs[i] = self.nodes[i](node_input)
        output = node_outputs[-1]
        output = self.output_proj(output)
        if self.use_classifier:
            output = self.attention_pooling(output)
            output = self.norm(output)
            output = self.head_mlp(output)
            return output
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, context):
        B, L, D = x.shape
        context_L = context.shape[1]
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, context_L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, context_L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out 