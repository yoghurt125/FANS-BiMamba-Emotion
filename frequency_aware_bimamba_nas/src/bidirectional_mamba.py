import os
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba module that processes sequences in both forward and backward directions
    then fuses the outputs using different fusion strategies.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, fusion_mode='projection'):
        super().__init__()
        self.d_model = d_model
        self.fusion_mode = fusion_mode
        
        # Forward and backward Mamba modules
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Fusion layers based on mode
        if fusion_mode == 'concat':
            self.output_dim = d_model * 2
            self.fusion_layer = None
        elif fusion_mode == 'add':
            self.output_dim = d_model
            self.fusion_layer = None
        elif fusion_mode == 'weighted':
            self.output_dim = d_model
            self.weight_forward = nn.Parameter(torch.tensor(0.5))
            self.weight_backward = nn.Parameter(torch.tensor(0.5))
            self.fusion_layer = None
        elif fusion_mode == 'projection':
            self.output_dim = d_model
            self.fusion_layer = nn.Linear(d_model * 2, d_model)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Fused output tensor
        """
        # Forward pass
        forward_out = self.forward_mamba(x)
        
        # Backward pass (reverse the sequence)
        x_reversed = torch.flip(x, dims=[1])
        backward_out = self.backward_mamba(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])  # Flip back to original order
        
        # Fusion
        if self.fusion_mode == 'concat':
            return torch.cat([forward_out, backward_out], dim=-1)
        elif self.fusion_mode == 'add':
            return forward_out + backward_out
        elif self.fusion_mode == 'weighted':
            return self.weight_forward * forward_out + self.weight_backward * backward_out
        elif self.fusion_mode == 'projection':
            concatenated = torch.cat([forward_out, backward_out], dim=-1)
            return self.fusion_layer(concatenated)


class BidirectionalMambaBlock(nn.Module):
    """
    A complete bidirectional Mamba block with normalization, activation, and dropout.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0, 
                 activation='relu', norm='none', fusion_mode='projection'):
        super().__init__()
        
        self.bidirectional_mamba = BidirectionalMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            fusion_mode=fusion_mode
        )
        
        # Store output dimension for downstream layers
        self.output_dim = self.bidirectional_mamba.output_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
        
        # Normalization layer
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(self.output_dim)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(self.output_dim)
        else:
            self.norm = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor after bidirectional Mamba processing
        """
        x = self.bidirectional_mamba(x)
        x = self.activation(x)
        
        # Handle batch normalization which expects (batch, features, sequence)
        if isinstance(self.norm, nn.BatchNorm1d):
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)
        
        x = self.dropout(x)
        return x