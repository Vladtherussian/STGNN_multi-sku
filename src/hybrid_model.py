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
        B, N, S, Feat = x.shape
        
        # --- Temporal Mixing ---
        x_temp = x.permute(0, 1, 3, 2)
        x_temp = self.temporal_mlp(x_temp)
        x_temp = x_temp.permute(0, 1, 3, 2)
        x = x + x_temp
        
        # --- Spatial Mixing (THE TRUE GCN FIX) ---
        # 1. Fold Time into the Batch dimension: [B, N, S, Feat] -> [B * S, N, Feat]
        x_folded = x.permute(0, 2, 1, 3).reshape(B * S, N, Feat)
        
        # 2. Duplicate the adjacency matrix for every time step
        adj_folded = adj.repeat_interleave(S, dim=0) # -> [B * S, N, N]
        
        # 3. Pass through the true PyTorch Geometric GCN layer!
        x_spatial_folded = self.spatial_gcn(x_folded, adj_folded)
        
        # 4. Unfold back to the original shape
        x_spatial = x_spatial_folded.view(B, S, N, Feat).permute(0, 2, 1, 3)
            
        # Add residual and normalize
        x = self.layer_norm(x + x_spatial)
        
        return x

class STGNNMixer(nn.Module):
    """
    Upgraded architecture featuring Local Instance Normalization 
    and Top-K Graph Sparsification.
    """
    def __init__(self, seq_len, pred_len, n_nodes, in_features, hidden_features, n_futr_features, n_blocks=3, top_k=5, ablation_mode="full"):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_nodes = n_nodes
        self.top_k = top_k # Restrict the graph to the top 3 strongest connections
        self.ablation_mode = ablation_mode # "full", "static_graph", or "no_graph"
        
        # Trainable identities
        self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden_features))
        if self.ablation_mode == "full":
            self.graph_alpha = nn.Parameter(torch.tensor(0.5))
        else:
            # For ablations, we freeze alpha so it cannot learn.
            # 1.0 means it relies 100% on the static Walmart hierarchy.
            self.register_buffer("graph_alpha", torch.tensor(1.0))
        
        # Instance Normalization across the temporal dimension
        # This acts as our lightweight version of RevIN to stabilize the gradients
        self.instance_norm = nn.InstanceNorm1d(in_features, affine=True)
        
        self.feature_projection = nn.Linear(in_features, hidden_features)
        # A learned pool to compress the 56 days without losing order
        self.context_pool = nn.Linear(seq_len, 1)
        
        # Instead of static node_emb, we will generate the graph dynamically
        self.query_proj = nn.Linear(hidden_features, hidden_features)
        self.key_proj = nn.Linear(hidden_features, hidden_features)
        
        # We can keep the static node embedding as a baseline "identity" for each node
        self.static_node_emb = nn.Parameter(torch.randn(n_nodes, hidden_features))

        # Add a heavy dropout to penalize transductive memorization
        self.emb_dropout = nn.Dropout(p=0.5)
        
        # We increased default blocks to 3 for deeper learning
        self.blocks = nn.ModuleList([
            GraphMixerBlock(seq_len=seq_len, n_features=hidden_features) 
            for _ in range(n_blocks)
        ])

        # Project the future features into the same hidden dimension
        self.futr_projection = nn.Linear(n_futr_features, hidden_features)

        # The final head now accepts (Historical Graph Memory) + (Future Info)
        self.head = nn.Sequential(
            nn.Linear((seq_len * hidden_features) + (pred_len * hidden_features), pred_len),
            nn.ReLU() # Stop negative predictions at the source!
        )
        

    def forward(self, x, x_future, adj):
        B, N, S, Feat = x.shape
        
        # 1. Local Normalization
        x_norm = x.view(B * N, S, Feat).permute(0, 2, 1) 
        x_norm = self.instance_norm(x_norm)
        x = x_norm.permute(0, 2, 1).view(B, N, S, Feat)
        
        # 2. Project features & add baseline static embeddings
        x = self.feature_projection(x) 
        # Apply dropout to the static ID embedding before adding it
        regularized_emb = self.emb_dropout(self.static_node_emb)
        x = x + regularized_emb.view(1, N, 1, -1)
        
        # ---------------------------------------------------------
        # 3. THE DYNAMIC GRAPH (Context-Aware Routing)
        # ---------------------------------------------------------
        if self.ablation_mode == "no_graph":
            combined_adj = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
            
        elif self.ablation_mode == "static_graph":
            combined_adj = F.normalize(adj, p=1, dim=1).unsqueeze(0).repeat(B, 1, 1)
            
        else: # "full" (Now completely dynamic!)
            # A. Summarize the 56-day history into a single context vector per node
            # Shape: [Batch, Nodes, Hidden]
            x_pool = x.permute(0, 1, 3, 2) # -> [B, N, Hidden, S]
            x_context = self.context_pool(x_pool).squeeze(-1) # -> [B, N, Hidden]
            
            # B. Generate Queries and Keys
            Q = self.query_proj(x_context) # [B, N, Hidden]
            K = self.key_proj(x_context)   # [B, N, Hidden]
            
            # C. Compute dynamic connection strength: Q * K^T
            # Shape: [B, N, N]
            dynamic_adj = torch.bmm(Q, K.transpose(1, 2))
            
            # D. Scale to prevent exploding gradients (standard attention mechanism)
            dynamic_adj = dynamic_adj / (Q.shape[-1] ** 0.5)
            dynamic_adj = F.relu(dynamic_adj)
            
            # E. Top-K Sparsification (Applied per batch!)
            topk_values, topk_indices = torch.topk(dynamic_adj, k=self.top_k, dim=2)
            mask = torch.zeros_like(dynamic_adj).scatter_(2, topk_indices, 1.0)
            
            # Replace zeros with -infinity before the softmax!
            dynamic_adj = dynamic_adj.masked_fill(mask == 0, float('-inf'))
            dynamic_adj = F.softmax(dynamic_adj, dim=2)

            # Constrain alpha strictly between 0.0 and 1.0
            alpha = torch.sigmoid(self.graph_alpha)
            
            # F. Blend with static business rules
            static_adj_batch = adj.unsqueeze(0).repeat(B, 1, 1)
            combined_adj = (alpha * static_adj_batch) + ((1 - alpha) * dynamic_adj)
            combined_adj = F.normalize(combined_adj, p=1, dim=2)

        # 4. Message Passing
        for block in self.blocks:
            x = block(x, combined_adj)
            
        x_hist_flat = x.reshape(B, N, S * x.shape[-1])
        
        # 5. Process Future Covariates
        x_futr_proj = self.futr_projection(x_future)
        x_futr_flat = x_futr_proj.reshape(B, N, -1)
        
        # 6. Predict
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
        # 1. Log Graph Alpha (Only if the core model has it!)
        if hasattr(self.model, 'graph_alpha'):
            alpha_val = self.model.graph_alpha.item()
            
            print(f"\n--- Epoch {self.current_epoch} End ---")
            print(f"Graph Alpha (Business vs. Latent): {alpha_val:.4f}")
            
            if alpha_val > 0.5:
                print("Status: Model is leaning on your Supply Chain Hierarchy.")
            else:
                print("Status: Model is leaning on Discovered Latent Relationships.")
            
            self.log("graph_alpha", alpha_val, prog_bar=True)

        # 2. Extract Top-K Learned Connections (Only if the model has dynamic node embeddings!)
        if hasattr(self.model, 'node_emb'):
            with torch.no_grad():
                learned_adj = torch.matmul(self.model.node_emb, self.model.node_emb.transpose(0, 1))
                learned_adj = F.relu(learned_adj)
                
                for i in range(min(3, self.model.n_nodes)):
                    weights, indices = torch.topk(learned_adj[i], k=5)
                    print(f"SKU {i} Strongest Learned Links: Nodes {indices.tolist()} (Weights: {weights.tolist()})")

class VanillaSTGNNBlock(nn.Module):
    """
    A classic Spatio-Temporal block using a GRU for time and GCN for space.
    """
    def __init__(self, n_features):
        super().__init__()
        
        # 1. Classic Temporal Processing (Recurrent Neural Network)
        self.temporal_gru = nn.GRU(
            input_size=n_features, 
            hidden_size=n_features, 
            batch_first=True
        )
        
        # 2. Classic Spatial Processing
        self.spatial_gcn = DenseGCNConv(in_channels=n_features, out_channels=n_features)
        
        self.layer_norm = nn.LayerNorm(n_features)

    def forward(self, x, adj):
        # x shape: [B, N, S, Feat]
        B, N, S, _ = x.shape
        
        # --- Temporal Processing (GRU) ---
        x_reshaped = x.view(B * N, S, -1)
        x_temp, _ = self.temporal_gru(x_reshaped) 
        x_temp = x_temp.view(B, N, S, -1)
        
        # --- Spatial Processing (GCN) ---
        x_proj = self.spatial_gcn.lin(x_temp)
        
        # The VRAM Saver
        x_proj_flat = x_proj.flatten(start_dim=2)
        x_spatial_flat = torch.bmm(adj, x_proj_flat)
        x_spatial = x_spatial_flat.view(B, N, S, -1)
        
        if getattr(self.spatial_gcn, 'bias', None) is not None:
            x_spatial = x_spatial + self.spatial_gcn.bias
            
        # Add residual and normalize
        x_out = self.layer_norm(x + x_spatial)
        return x_out


class VanillaSTGNN(nn.Module):
    """The full model wrapper for the Vanilla baseline."""
    # NEW: Added n_futr_features to the init arguments
    def __init__(self, seq_len, pred_len, n_nodes, in_features, hidden_features, n_blocks, n_futr_features):
        super().__init__()
        
        # 1. Feature Projection
        self.input_proj = nn.Linear(in_features, hidden_features)
        
        # 2. The Deep Layers
        self.blocks = nn.ModuleList([
            VanillaSTGNNBlock(n_features=hidden_features) 
            for _ in range(n_blocks)
        ])
        
        # NEW: 3. Project future features into the hidden dimension
        self.futr_projection = nn.Linear(n_futr_features, hidden_features)
        
        # NEW: 4. The Output Head now accepts (GRU Final State) + (Flattened Future Info)
        # Also added the ReLU to stop negative predictions!
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_features + (pred_len * hidden_features), pred_len),
            nn.ReLU() 
        )

    def forward(self, x, x_future, adj):
        B, N, S, _ = x.shape
        
        x = self.input_proj(x)
        
        import torch.nn.functional as F
        adj_norm = F.normalize(adj, p=1, dim=1)
        adj_batched = adj_norm.unsqueeze(0).repeat(B, 1, 1)
        
        for block in self.blocks:
            x = block(x, adj_batched)
            
        # The GRU's summary of the 56-day history: [B, N, Hidden]
        x_final = x[:, :, -1, :] 
        
        # NEW: Process Future Covariates
        x_futr_proj = self.futr_projection(x_future)         # [B, N, Pred_Len, Hidden]
        x_futr_flat = x_futr_proj.reshape(B, N, -1)          # [B, N, Pred_Len * Hidden]
        
        # NEW: Combine Historical Context with Future Knowledge
        combined = torch.cat([x_final, x_futr_flat], dim=2)  # [B, N, Hidden + (Pred_Len * Hidden)]
        
        out = self.out_proj(combined)
        return out