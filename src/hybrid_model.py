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
        
        # --- Spatial Mixing (Optimized Parallel Graph Convolution) ---
        # 1. Swap Seq_len and Nodes -> [Batch, Seq_len, Nodes, Features]
        x_reshaped = x.permute(0, 2, 1, 3) 
        
        # 2. Flatten Batch and Seq_len into one massive effective batch
        # Shape becomes: [(Batch * Seq_len), Nodes, Features]
        x_flat = x_reshaped.reshape(B * S, N, Feat)
        
        # 3. Apply the Graph Convolution ONCE across all time steps simultaneously!
        # PyTorch automatically broadcasts your [N, N] adjacency matrix across the new batch size.
        gcn_out = self.spatial_gcn(x_flat, adj) 
        
        # 4. Unflatten back to the original dimensions
        # View separates the Batch and Time, Permute puts Nodes back in the right spot
        x_spatial = gcn_out.view(B, S, N, Feat).permute(0, 2, 1, 3) # -> [B, N, S, Feat]
        
        # Add residual and normalize
        x = self.layer_norm(x + x_spatial)
        
        return x

class STGNNMixer(nn.Module):
    """
    Upgraded architecture featuring Local Instance Normalization 
    and Top-K Graph Sparsification.
    """
    def __init__(self, seq_len, pred_len, n_nodes, in_features, hidden_features, n_futr_features, n_blocks=3, top_k=10):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_nodes = n_nodes
        self.top_k = top_k # Restrict the graph to the top 10 strongest connections
        
        # Trainable identities
        self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden_features))
        self.graph_alpha = nn.Parameter(torch.tensor(0.5))
        
        # Instance Normalization across the temporal dimension
        # This acts as our lightweight version of RevIN to stabilize the gradients
        self.instance_norm = nn.InstanceNorm1d(in_features, affine=True)
        
        self.feature_projection = nn.Linear(in_features, hidden_features)
        
        # We increased default blocks to 3 for deeper learning
        self.blocks = nn.ModuleList([
            GraphMixerBlock(seq_len=seq_len, n_features=hidden_features) 
            for _ in range(n_blocks)
        ])

        # Project the future features into the same hidden dimension
        self.futr_projection = nn.Linear(n_futr_features, hidden_features)

        # The final head now accepts (Historical Graph Memory) + (Future Info)
        self.head = nn.Linear((seq_len * hidden_features) + (pred_len * hidden_features), pred_len)
        

    def forward(self, x, x_future, adj):
        B, N, S, Feat = x.shape
        
        # ---------------------------------------------------------
        # 1. Local Normalization (Crucial for Time Series)
        # ---------------------------------------------------------
        # InstanceNorm1d expects [Batch, Features, Time]. We reshape, normalize, and revert.
        x_norm = x.view(B * N, S, Feat).permute(0, 2, 1) 
        x_norm = self.instance_norm(x_norm)
        x = x_norm.permute(0, 2, 1).view(B, N, S, Feat)
        
        # 2. Project features & add embeddings
        x = self.feature_projection(x) 
        x = x + self.node_emb.view(1, N, 1, -1)
        
        # ---------------------------------------------------------
        # 3. Top-K Sparsified Adaptive Graph
        # ---------------------------------------------------------
        learned_adj = torch.matmul(self.node_emb, self.node_emb.transpose(0, 1))
        learned_adj = F.relu(learned_adj)
        
        # Force sparsity. Find the top K connections for each node.
        # This prevents over-smoothing and creates sharp, interpretable edges.
        topk_values, topk_indices = torch.topk(learned_adj, k=self.top_k, dim=1)
        mask = torch.zeros_like(learned_adj).scatter_(1, topk_indices, 1.0)
        
        # Blending static business rules with learned insights.
        # graph_alpha is a trainable parameter.
        # If the optimizer increases alpha, it favors your hierarchy. 
        # If it decreases alpha, it favors discovered product correlations
        learned_adj = F.softmax(learned_adj * mask, dim=1)
        combined_adj = (self.graph_alpha * adj) + ((1 - self.graph_alpha) * learned_adj)
        combined_adj = F.normalize(combined_adj, p=1, dim=1)
        
        for block in self.blocks:
            x = block(x, combined_adj)
            
        x_hist_flat = x.reshape(B, N, S * x.shape[-1]) # [Batch, Nodes, 56 * Hidden]
        
        # NEW: Process Future Covariates
        x_futr_proj = self.futr_projection(x_future)   # [Batch, Nodes, 28, Hidden]
        x_futr_flat = x_futr_proj.reshape(B, N, -1)    # [Batch, Nodes, 28 * Hidden]
        
        # Concatenate History and Future, then predict!
        combined = torch.cat([x_hist_flat, x_futr_flat], dim=2)
        out = self.head(combined) 
        return out


class GraphTimeSeriesDataset(Dataset):
    """Slices the 3D tensors into sliding temporal windows for training."""
    def __init__(self, X, y, seq_len, pred_len, futr_indices):
        # X shape: [Nodes, Time, Features]
        # y shape: [Nodes, Time]
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.futr_indices = futr_indices
        # Calculate how many valid sliding windows we can make
        self.valid_starts = X.shape[1] - seq_len - pred_len + 1

    def __len__(self):
        return self.valid_starts

    def __getitem__(self, idx):
        # Extract a window for ALL 80 nodes simultaneously
        x_window = self.X[:, idx : idx + self.seq_len, :]
        y_hist = self.y[:, idx : idx + self.seq_len] # Used to calculate RevIN mean/std
        x_future = self.X[:, idx + self.seq_len : idx + self.seq_len + self.pred_len, self.futr_indices]
        y_window = self.y[:, idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x_window, y_hist, x_future, y_window


class LitSTGNNMixer(pl.LightningModule):
    """The PyTorch Lightning wrapper that handles the training loop and GPU acceleration."""
    def __init__(self, model, adj_matrix, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        # Register the adjacency matrix as a PyTorch buffer
        self.register_buffer("adj_matrix", adj_matrix)

    def forward(self, x, x_future):
        return self.model(x, x_future, self.adj_matrix)

    def training_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        # 1. Calculate RevIN statistics from the HISTORY
        mean = y_hist.mean(dim=-1, keepdim=True)
        std = y_hist.std(dim=-1, keepdim=True) + 1e-5
        
        # 2. Normalize the Target
        y_norm = (y - mean) / std
        
        # 3. Predict the normalized future
        y_hat_norm = self(x, x_future)
        
        loss = F.l1_loss(y_hat_norm, y_norm)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        mean = y_hist.mean(dim=-1, keepdim=True)
        std = y_hist.std(dim=-1, keepdim=True) + 1e-5
        
        y_hat_norm = self(x, x_future)
        
        # 4. REVERSE the normalization back to real-world sales volume
        y_hat = (y_hat_norm * std) + mean
        
        # Calculate loss against the RAW target to get true MAE
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        mean = y_hist.mean(dim=-1, keepdim=True)
        std = y_hist.std(dim=-1, keepdim=True) + 1e-5
        
        y_hat_norm = self(x, x_future)
        y_hat = (y_hat_norm * std) + mean
        
        # Calculate loss against the RAW target
        loss = F.l1_loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_train_epoch_end(self):
        # Access the alpha from the core model
        # .item() converts the 1-element tensor to a regular Python float
        alpha_val = self.model.graph_alpha.item()
        
        print(f"\n--- Epoch {self.current_epoch} End ---")
        print(f"Graph Alpha (Business vs. Latent): {alpha_val:.4f}")
        
        # Interpret the result for your research log
        if alpha_val > 0.5:
            print("Status: Model is leaning on your Supply Chain Hierarchy.")
        else:
            print("Status: Model is leaning on Discovered Latent Relationships.")
        
        # Also log it so it shows up in your progress bar/tensorboard
        self.log("graph_alpha", alpha_val, prog_bar=True)

        # 2. Extract Top-K Learned Connections
        # We look at the learned adjacency matrix from the model
        with torch.no_grad():
            # Recompute learned_adj just for the printout
            learned_adj = torch.matmul(self.model.node_emb, self.model.node_emb.transpose(0, 1))
            learned_adj = F.relu(learned_adj)
            
            # Let's look at the first 3 products as a sample
            for i in range(min(3, self.model.n_nodes)):
                weights, indices = torch.topk(learned_adj[i], k=5)
                print(f"SKU {i} Strongest Learned Links: Nodes {indices.tolist()} (Weights: {weights.tolist()})")