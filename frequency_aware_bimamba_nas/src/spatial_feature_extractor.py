import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGGraphStructure:
    def __init__(self):
        self.channel_names = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'C4',
            'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Cz'
        ]
        
        self.positions = {
            'Fp1': (-0.309, 0.951), 'AF3': (-0.545, 0.839), 'F3': (-0.707, 0.707),
            'F7': (-0.951, 0.309), 'FC5': (-0.891, 0.454), 'FC1': (-0.454, 0.891),
            'C3': (-1.0, 0.0), 'T7': (-0.951, -0.309), 'CP5': (-0.891, -0.454),
            'CP1': (-0.454, -0.891), 'P3': (-0.707, -0.707), 'P7': (-0.951, -0.588),
            'PO3': (-0.545, -0.839), 'O1': (-0.309, -0.951), 'Oz': (0.0, -1.0),
            'Pz': (0.0, -0.707), 'Fp2': (0.309, 0.951), 'AF4': (0.545, 0.839),
            'Fz': (0.0, 1.0), 'F4': (0.707, 0.707), 'F8': (0.951, 0.309),
            'FC6': (0.891, 0.454), 'FC2': (0.454, 0.891), 'C4': (1.0, 0.0),
            'T8': (0.951, -0.309), 'CP6': (0.891, -0.454), 'CP2': (0.454, -0.891),
            'P4': (0.707, -0.707), 'P8': (0.951, -0.588), 'PO4': (0.545, -0.839),
            'O2': (0.309, -0.951), 'Cz': (0.0, 0.0)
        }
    
    def get_adjacency_matrix(self, threshold=0.4):
        n_channels = len(self.channel_names)
        adj_matrix = np.zeros((n_channels, n_channels))
        
        for i, ch1 in enumerate(self.channel_names):
            for j, ch2 in enumerate(self.channel_names):
                if i != j:
                    pos1 = np.array(self.positions[ch1])
                    pos2 = np.array(self.positions[ch2])
                    distance = np.linalg.norm(pos1 - pos2)
                    if distance < threshold:
                        adj_matrix[i, j] = 1.0
                else:
                    adj_matrix[i, j] = 1.0
        
        return adj_matrix

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        adj_expanded = adj.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.bmm(adj_expanded, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class EEGGraphConvNet(nn.Module):
    def __init__(self, n_channels, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        graph_structure = EEGGraphStructure()
        adj_matrix = graph_structure.get_adjacency_matrix()
        
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        degree_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix + 1e-6))
        normalized_adj = degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt
        
        self.register_buffer('adj_matrix', torch.FloatTensor(normalized_adj))
        
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.gcn_layers.append(BatchGraphConvLayer(1, hidden_dim))
            else:
                self.gcn_layers.append(BatchGraphConvLayer(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        batch_size, n_channels, time_length = x.shape
        
        x_transposed = x.permute(0, 2, 1).contiguous()
        node_feat = x_transposed.reshape(batch_size * time_length, n_channels, 1)
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            node_feat = gcn_layer(node_feat, self.adj_matrix)
            
            if i < len(self.gcn_layers) - 1:
                node_feat = self.activation(node_feat)
                node_feat = self.dropout(node_feat)
        
        result = node_feat.reshape(batch_size, time_length, n_channels, self.hidden_dim)
        result = result.permute(0, 2, 3, 1).contiguous()
        result = result.reshape(batch_size, n_channels * self.hidden_dim, time_length)
        
        return result

class SpatialAttentionModule(nn.Module):
    def __init__(self, input_dim, n_channels=32, reduction_ratio=4):
        super().__init__()
        self.n_channels = n_channels
        self.channel_dim = input_dim // n_channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(input_dim, input_dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(input_dim // reduction_ratio, input_dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for electrode positions
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(input_dim // reduction_ratio, n_channels, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Conv1d(input_dim, input_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim, time_length)
        batch_size, input_dim, time_length = x.shape
        
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_attention(x_channel)  # (batch_size, n_channels, time_length)
        
        # Reshape input for spatial attention application
        x_reshaped = x_channel.view(batch_size, self.n_channels, self.channel_dim, time_length)
        spatial_att_expanded = spatial_att.unsqueeze(2)  # (batch_size, n_channels, 1, time_length)
        
        # Apply spatial attention
        x_spatial = x_reshaped * spatial_att_expanded
        x_spatial = x_spatial.view(batch_size, input_dim, time_length)
        
        # Feature transformation
        output = self.feature_transform(x_spatial)
        
        # Residual connection
        return output + x

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, n_channels=32, gcn_hidden_dim=8, gcn_layers=2, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.gcn_hidden_dim = gcn_hidden_dim
        
        # GCN for spatial structure learning
        self.gcn = EEGGraphConvNet(
            n_channels=n_channels,
            hidden_dim=gcn_hidden_dim,
            n_layers=gcn_layers,
            dropout=dropout
        )
        
        # Spatial attention module
        gcn_output_dim = n_channels * gcn_hidden_dim
        self.spatial_attention = SpatialAttentionModule(
            input_dim=gcn_output_dim,
            n_channels=n_channels
        )
        
        # Output projection to maintain original channel dimension
        self.output_proj = nn.Conv1d(gcn_output_dim, n_channels, 1)
        
        # Normalization and activation
        self.norm = nn.BatchNorm1d(n_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, channels, time_length)
        # Step 1: GCN for spatial structure
        x_gcn = self.gcn(x)  # (batch_size, n_channels * gcn_hidden_dim, time_length)
        
        # Step 2: Spatial attention
        x_attention = self.spatial_attention(x_gcn)
        
        # Step 3: Project back to original dimension
        x_proj = self.output_proj(x_attention)  # (batch_size, n_channels, time_length)
        
        # Step 4: Normalization and activation
        x_norm = self.norm(x_proj)
        x_out = self.activation(x_norm)
        x_out = self.dropout(x_out)
        
        # Residual connection with input
        return x_out + x