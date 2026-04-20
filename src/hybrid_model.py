import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F

class GraphMixerBlock(nn.Module):
    """
    The core innovation: A hybrid block that mixes temporal patterns via MLP, 
    and spatial patterns via Graph Convolutions.
    """
    def __init__(self, seq_len, n_features, dropout=0.2):
        super().__init__()
        
        # 1. Temporal Mixing (The MLP part)
        # Looks across time for a single SKU to find trends and lag patterns
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Spatial Mixing (The GNN part)
        # Replaces the dense MLP with an explicit Graph Convolution
        self.spatial_gcn = DenseGCNConv(in_channels=n_features, out_channels=n_features)
        
        self.layer_norm = nn.LayerNorm(n_features)

    def forward(self, x, adj):
        # x shape: [Batch_size, Num_nodes, Seq_len, Features]
        B, N, S, Feat = x.shape
        
        # --- Temporal Mixing ---
        # Reshape to mix across the sequence length (time)
        x_temp = x.permute(0, 1, 3, 2) # [B, N, Feat, S]
        x_temp = self.temporal_mlp(x_temp)
        x_temp = x_temp.permute(0, 1, 3, 2) # Back to [B, N, S, Feat]
        
        # Add residual connection
        x = x + x_temp
        
        # --- Spatial Mixing (Graph Convolution) ---
        # DenseGCNConv expects [Batch, Nodes, Features]. 
        # We loop through the time steps (Seq_len) to apply the graph across the features at each day.
        out_spatial = []
        for t in range(S):
            # Extract the graph state at time t: shape [B, N, Feat]
            x_t = x[:, :, t, :] 
            
            # Pass through the Graph Convolution using your adjacency matrix!
            gcn_out = self.spatial_gcn(x_t, adj)
            out_spatial.append(gcn_out.unsqueeze(2))
            
        # Recombine time steps
        x_spatial = torch.cat(out_spatial, dim=2) # [B, N, S, Feat]
        
        # Add residual and normalize
        x = self.layer_norm(x + x_spatial)
        
        return x

class STGNNMixer(nn.Module):
    """
    The complete PyTorch architecture with Trainable Spatial Node Embeddings
    and an Adaptive Latent Adjacency Matrix.
    """
    def __init__(self, seq_len, pred_len, n_nodes, in_features, hidden_features, n_blocks=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_nodes = n_nodes
        
        # This gives each SKU a unique trainable identity
        self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden_features))

        # A trainable parameter that learns how to balance the 
        # hardcoded business rules vs. the data-driven discovered relationships.
        # Initialized at 0.5 (50/50 split)
        self.graph_alpha = nn.Parameter(torch.tensor(0.5))
        
        # Project raw features into a richer hidden dimension
        self.feature_projection = nn.Linear(in_features, hidden_features)
        
        # Stack our custom Hybrid Blocks
        self.blocks = nn.ModuleList([
            GraphMixerBlock(seq_len=seq_len, n_features=hidden_features) 
            for _ in range(n_blocks)
        ])
        
        # Final output layer to map the hidden features down to the prediction horizon
        self.head = nn.Linear(seq_len * hidden_features, pred_len)

    def forward(self, x, adj):
        # x shape: [Batch, Nodes, Seq_len, Features]
        B, N, S, Feat = x.shape
        
        # 1. Project features
        x = self.feature_projection(x) # [B, N, S, Hidden]
        
        # 2. Add spatial node identity (Broadcasting across Batch and Seq_len dimensions)
        # self.node_emb is [N, Hidden]. We reshape it to [1, N, 1, Hidden] so it adds perfectly.
        x = x + self.node_emb.view(1, N, 1, -1)
        
        # 3. Compute the Adaptive Adjacency Matrix
        # Multiply the embeddings [N, Hidden] by transpose [Hidden, N] to get [N, N]
        learned_adj = torch.matmul(self.node_emb, self.node_emb.transpose(0, 1))
        
        # ReLU removes negative connections, Softmax normalizes the edge weights
        learned_adj = F.softmax(F.relu(learned_adj), dim=1)
        
        # Blend the explicit business graph (adj) with the implicit learned graph
        # using the trainable alpha parameter.
        combined_adj = (self.graph_alpha * adj) + ((1 - self.graph_alpha) * learned_adj)
        
        # 4. Pass through the hybrid blocks using the blended graph!
        for block in self.blocks:
            x = block(x, combined_adj)
            
        # 5. Flatten time and features to predict the future
        x = x.reshape(B, N, S * x.shape[-1])
        
        # 6. Output predictions for the next 28 days
        out = self.head(x) # [Batch, Nodes, Pred_len]
        return out


class GraphTimeSeriesDataset(Dataset):
    """Slices the 3D tensors into sliding temporal windows for training."""
    def __init__(self, X, y, seq_len, pred_len):
        # X shape: [Nodes, Time, Features]
        # y shape: [Nodes, Time]
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Calculate how many valid sliding windows we can make
        self.valid_starts = X.shape[1] - seq_len - pred_len + 1

    def __len__(self):
        return self.valid_starts

    def __getitem__(self, idx):
        # Extract a window for ALL 80 nodes simultaneously
        x_window = self.X[:, idx : idx + self.seq_len, :]
        y_window = self.y[:, idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x_window, y_window


class LitSTGNNMixer(pl.LightningModule):
    """The PyTorch Lightning wrapper that handles the training loop and GPU acceleration."""
    def __init__(self, model, adj_matrix, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        # Register the adjacency matrix as a PyTorch buffer
        self.register_buffer("adj_matrix", adj_matrix)

    def forward(self, x):
        return self.model(x, self.adj_matrix)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        # We use L1 Loss (Mean Absolute Error) to match NeuralForecast's baseline
        loss = F.l1_loss(y_hat, y) 
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)