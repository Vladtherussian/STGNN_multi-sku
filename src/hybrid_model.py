import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv  # used by VanillaSTGNNBlock only
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F

class GraphMixerBlock(nn.Module):
    """
    Upgraded to Pre-Norm Architecture for FP16 stability.
    Applies LayerNorm before every sub-block to prevent activation overflow.
    """
    def __init__(self, seq_len, n_features, dropout=0.2):
        super().__init__()
        self.weekly_conv  = nn.Conv1d(1, 1, kernel_size=7, padding="same", bias=False)
        self.monthly_conv = nn.Conv1d(1, 1, kernel_size=28, padding="same", bias=False)

        # Pre-Norm layers for every residual pathway
        self.norm_conv = nn.LayerNorm(n_features)
        self.norm_temp = nn.LayerNorm(n_features)
        self.norm_feat = nn.LayerNorm(n_features)
        self.norm_spat = nn.LayerNorm(n_features)

        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.spatial_lin  = nn.Linear(n_features, n_features, bias=False)
        self.spatial_bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, adj):
        B, N, S, Feat = x.shape

        # 1. Convolution Mixing (Pre-Norm)
        x_norm = self.norm_conv(x)
        x_conv = x_norm.permute(0, 1, 3, 2).reshape(B * N * Feat, 1, S)
        weekly_signal  = self.weekly_conv(x_conv).reshape(B, N, Feat, S).permute(0, 1, 3, 2)
        monthly_signal = self.monthly_conv(x_conv).reshape(B, N, Feat, S).permute(0, 1, 3, 2)
        x = x + weekly_signal + monthly_signal   
        
        # 2. Temporal Mixing (Pre-Norm)
        x_norm = self.norm_temp(x)
        x_temp = x_norm.permute(0, 1, 3, 2)          
        x_temp = self.temporal_mlp(x_temp)
        x_temp = x_temp.permute(0, 1, 3, 2)      
        x = x + x_temp

        # 3. Feature Mixing (Pre-Norm)
        x_norm = self.norm_feat(x)
        x_feat = self.feature_mlp(x_norm)
        x = x + x_feat
        
        # 4. Spatial Mixing (Pre-Norm)
        x_norm = self.norm_spat(x)
        x_proj = self.spatial_lin(x_norm)
        
        x_proj_2d = x_proj.reshape(B, N, S * Feat)
        x_spatial_2d = torch.bmm(adj, x_proj_2d)
        x_spatial = x_spatial_2d.reshape(B, N, S, Feat) + self.spatial_bias
            
        x = x + x_spatial
        
        return x

class STGNNMixer(nn.Module):
    """
    Upgraded architecture featuring Local Instance Normalization 
    and Top-K Graph Sparsification.

    RTX 3080 (10 GB) recommended defaults
    ──────────────────────────────────────
    hidden_features : 64    (was 128 — halving cuts most tensors by 4x)
    n_blocks        : 2     (was 3  — saves one block's activation budget)
    batch_size      : 8     (was 32 — set in assets.py / train asset)

    The dominant VRAM consumers at these settings (N=1437, S=56, fp16):
      combined_adj  [B, N, N]         :  ~24 MB
      feature maps  [B, N, S, Feat]   : ~105 MB  (per block, before ckpt)
      x_proj_2d     [B, N, S*Feat]    : ~105 MB  (peak inside block)
    Total estimated peak: ~2–3 GB, leaving headroom for optimizer states.
    """
    def __init__(self, seq_len, pred_len, n_nodes, in_features, hidden_features, n_futr_features, n_blocks=2, top_k=5, ablation_mode="full"):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_nodes = n_nodes
        self.top_k = top_k # Restrict the graph to the top 3 strongest connections
        self.ablation_mode = ablation_mode # "full", "static_graph", or "no_graph"
        
        # Trainable identities
        # self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden_features))
        if self.ablation_mode == "full":
            self.graph_alpha = nn.Parameter(torch.tensor(2.20)) # Start with a bias towards the static graph (sigmoid(2.2) ~ 0.9)
        else:
            # For ablations, we freeze alpha so it cannot learn.
            # 1.0 means it relies 100% on the static Walmart hierarchy.
            self.register_buffer("graph_alpha", torch.tensor(1.0))
        
        
        self.feature_projection = nn.Linear(in_features, hidden_features)
        self.input_norm = nn.LayerNorm(hidden_features)
        
        # --- UPGRADE 1: Exponential Recency Weighting ---
        # Exponentially decaying weights — more weight on recent time steps 
        # 14.0 is the half-life: recent promotional behavior dominates the graph construction [cite: 121, 122]
        decay = torch.exp(-torch.arange(seq_len, 0, -1).float() / 14.0)
        self.register_buffer('recency_weights', decay / decay.sum())
        
        # --- UPGRADE 2: Zero-Inflation Gate ---
        # A Bernoulli gate to predict "is this item selling at all during this horizon" [cite: 131]
        # self.zero_gate = nn.Linear((seq_len * hidden_features) + (pred_len * hidden_features), pred_len)
        
        # Instead of static node_emb, we will generate the graph dynamically
        self.query_proj = nn.Linear(hidden_features, hidden_features)
        self.key_proj = nn.Linear(hidden_features, hidden_features)
        
        # We can keep the static node embedding as a baseline "identity" for each node
        self.static_node_emb = nn.Parameter(torch.randn(n_nodes, hidden_features) * 0.02) # Small init to prevent early training instability

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
        self.head = nn.Sequential(nn.Linear((seq_len * hidden_features) + (pred_len * hidden_features), pred_len))
        

    def forward(self, x, x_future, adj):
        B, N, S, Feat = x.shape
        
        # 1. Local Normalization
        # x_norm = x.view(B * N, S, Feat).permute(0, 2, 1) 
        # x_norm = self.instance_norm(x_norm)
        # x = x_norm.permute(0, 2, 1).view(B, N, S, Feat)
        
        # 2. Project features & add baseline static embeddings
        x = self.feature_projection(x) 
        x = self.input_norm(x)
        # Apply dropout to the static ID embedding before adding it
        regularized_emb = self.emb_dropout(self.static_node_emb)
        regularized_emb = F.normalize(regularized_emb, p=2, dim=-1)  # unit norm
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
            # Shape: [B, N, Hidden]
            # We use the exponential recency weights so recent shocks drive the graph 
            x_context = (x * self.recency_weights.view(1, 1, -1, 1)).sum(dim=2)
            
            # B. Generate Queries and Keys
            Q = self.query_proj(x_context) # [B, N, Hidden]
            K = self.key_proj(x_context)   # [B, N, Hidden]
            
            # C. Compute dynamic connection strength: Q * K^T
            # Shape: [B, N, N]
            dynamic_adj = torch.bmm(Q, K.transpose(1, 2))
            
            # D. Scale to prevent exploding gradients (standard attention mechanism)
            dynamic_adj = dynamic_adj / (Q.shape[-1] ** 0.5)
            
            # Guard 1: clamp scores before ReLU.
            # Under fp16 mixed precision, large Q/K dot products can overflow to +inf.
            # +inf survives ReLU and then causes softmax([+inf, -inf, ...]) = NaN.
            # Clamping to [-10, 10] is safe — softmax is saturated well before these values.
            dynamic_adj = dynamic_adj.clamp(min=-10.0, max=10.0)
            dynamic_adj = F.relu(dynamic_adj)
            
            # E. Top-K Sparsification (Applied per batch!)
            # Guard 2: top_k must not exceed N. If the downsampling config changes and
            # produces fewer nodes than top_k, torch.topk raises a hard runtime error.
            safe_k = min(self.top_k, N)
            topk_values, topk_indices = torch.topk(dynamic_adj, k=safe_k, dim=2)
            mask = torch.zeros_like(dynamic_adj).scatter_(2, topk_indices, 1.0)
            
            # Replace non-top-K positions with -infinity before the softmax.
            dynamic_adj = dynamic_adj.masked_fill(mask == 0, float('-inf'))
            dynamic_adj = F.softmax(dynamic_adj, dim=2)
            
            # Guard 3: nan_to_num as a final safety net.
            # If an entire row was zero after ReLU AND all top-K positions are zero,
            # softmax still produces a valid uniform distribution (not NaN). But if any
            # upstream NaN slips through (e.g. from gradient tape in rare fp16 edge cases),
            # this replaces it with 0 so the row gets no graph signal rather than poisoning
            # all downstream gradients with NaN.
            dynamic_adj = torch.nan_to_num(dynamic_adj, nan=0.0)

            # Constrain alpha strictly between 0.0 and 1.0
            alpha = torch.sigmoid(self.graph_alpha)
            
            # F. Blend with static business rules
            static_adj_batch = adj.unsqueeze(0).repeat(B, 1, 1)
            combined_adj = (alpha * static_adj_batch) + ((1 - alpha) * dynamic_adj)
            combined_adj = F.normalize(combined_adj, p=1, dim=2)

        # 4. Message Passing with gradient checkpointing.
        # Each GraphMixerBlock stores [B, N, S, Feat] activations for backprop.
        # At B=8, N=1437, S=56, Feat=64 that is ~210 MB per block in fp32.
        # Checkpointing discards those activations and recomputes them during
        # the backward pass, saving ~(n_blocks - 1) * block_activation_memory
        # at the cost of one extra forward pass per block.
        from torch.utils.checkpoint import checkpoint
        for block in self.blocks:
            x = checkpoint(block, x, combined_adj, use_reentrant=False)
            
        x_hist_flat = x.reshape(B, N, S * x.shape[-1])
        
        # 5. Process Future Covariates
        x_futr_proj = self.futr_projection(x_future)
        x_futr_flat = x_futr_proj.reshape(B, N, -1)
        
        # 6. Predict (No Gate!)
        combined = torch.cat([x_hist_flat, x_futr_flat], dim=2)
        sales_forecast = self.head(combined)                        
        
        return sales_forecast


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
        self.register_buffer("adj_matrix", adj_matrix)

    def forward(self, x, x_future):
        return self.model(x, x_future, self.adj_matrix)

    def training_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        # --- FP32 Armor ---
        y_hist_fp32 = y_hist.float()
        y_fp32 = y.float()
        
        mean = y_hist_fp32.mean(dim=-1, keepdim=True)
        var = y_hist_fp32.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        
        # Forward pass & Softplus
        y_hat_norm = self(x, x_future)
        y_hat_norm = y_hat_norm.float()
        
        y_hat_raw = (y_hat_norm * std) + mean
        y_hat = F.softplus(y_hat_raw) 
        
        y_clamp = y_fp32.clamp(min=0.0)

        # ==========================================
        # 1. THE FIX: Poisson Loss for the Primary Objective
        # ==========================================
        # log_input=False tells PyTorch our network is already outputting raw positive values (thanks to Softplus)
        primary_loss = F.poisson_nll_loss(y_hat, y_clamp, log_input=False)
        
        # 2. Graph Consistency Loss (Kept in Log Space for Magnitude Stability)
        # We still use log1p here strictly so the magnitude of this regularization 
        # doesn't explode and ruin the 0.1 scaling weight.
        log_y = torch.log1p(y_clamp)
        log_y_hat = torch.log1p(y_hat)
        
        residuals = log_y - log_y_hat          
        res_mean = residuals.mean(dim=-1)          
        
        neighbor_res = torch.bmm(
            self.adj_matrix.float().unsqueeze(0).expand(residuals.shape[0], -1, -1),
            res_mean.unsqueeze(-1)
        ).squeeze(-1)
        
        graph_loss = F.mse_loss(res_mean, neighbor_res.detach())
        loss = primary_loss + (0.1 * graph_loss)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('graph_consistency_loss', graph_loss, prog_bar=False)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        # --- FP32 ARMOR ---
        y_hist_fp32 = y_hist.float()
        y_fp32 = y.float()
        
        mean = y_hist_fp32.mean(dim=-1, keepdim=True)
        var = y_hist_fp32.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        
        y_hat_norm = self(x, x_future)
        y_hat_norm = y_hat_norm.float()
        
        y_hat_raw = (y_hat_norm * std) + mean
        y_hat = F.softplus(y_hat_raw) 
        
        y_clamp = y_fp32.clamp(min=0.0)

        # Track Poisson Loss in Validation as well
        loss = F.poisson_nll_loss(y_hat, y_clamp, log_input=False)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        # --- FP32 ARMOR ---
        y_hist_fp32 = y_hist.float()
        y_fp32 = y.float()
        
        mean = y_hist_fp32.mean(dim=-1, keepdim=True)
        var = y_hist_fp32.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        
        y_hat_norm = self(x, x_future)
        y_hat_norm = y_hat_norm.float()
        
        y_hat_raw = (y_hat_norm * std) + mean
        y_hat = F.softplus(y_hat_raw) 
        
        y_clamp = y_fp32.clamp(min=0.0)

        # Track Poisson Loss in Test as well
        loss = F.poisson_nll_loss(y_hat, y_clamp, log_input=False)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_train_epoch_end(self):
        if hasattr(self.model, 'graph_alpha'):
            alpha_val = torch.sigmoid(self.model.graph_alpha).item()
            print(f"\n--- Epoch {self.current_epoch} End ---")
            print(f"Graph Alpha (Business vs. Latent): {alpha_val:.4f}")
            self.log("graph_alpha", alpha_val, prog_bar=True)

        if hasattr(self.model, 'static_node_emb'):
            with torch.no_grad():
                learned_adj = torch.matmul(self.model.static_node_emb, self.model.static_node_emb.transpose(0, 1))
                learned_adj = F.relu(learned_adj)
                for i in range(min(3, self.model.n_nodes)):
                    weights, indices = torch.topk(learned_adj[i], k=5)
                    print(f"SKU {i} Strongest Learned Links: Nodes {indices.tolist()} (Weights: {weights.tolist()})")


class LitResidualSTGNN(pl.LightningModule):
    """
    Dedicated Lightning wrapper for the Residual Two-Stage Model. 
    Now features a self-learning shrinkage parameter to automatically 
    calibrate the magnitude of the spatial corrections!
    """
    def __init__(self, model, adj_matrix, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.register_buffer("adj_matrix", adj_matrix)
        
        # THE ELEGANT FIX: A learnable parameter for the residual scale.
        # We initialize it at 0.1 so the model starts cautious, 
        # but the optimizer will push this up or down automatically!
        self.learned_shrinkage = nn.Parameter(torch.ones(1, model.n_nodes, 1) * 0.1)

    def forward(self, pure_x, x_future):
        # The core model predicts the raw spatial shock
        raw_residual = self.model(pure_x, x_future, self.adj_matrix)
        
        # The network scales its own prediction before outputting it!
        return raw_residual * self.learned_shrinkage

    def training_step(self, batch, _batch_idx):
        x, y_hist, x_future, y_residual = batch
        
        # The Pure Residual Hack
        pure_x = y_hist.unsqueeze(-1) 
        
        # Forward pass (Now natively scaled by the learned shrinkage!)
        y_hat_residual = self(pure_x, x_future)
        
        loss = F.mse_loss(y_hat_residual, y_residual)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y_hist, x_future, y_residual = batch
        pure_x = y_hist.unsqueeze(-1)
        
        y_hat_residual = self(pure_x, x_future)
        loss = F.mse_loss(y_hat_residual, y_residual)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # We still keep a tiny bit of weight decay to keep the core graph weights healthy
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
    def on_train_epoch_end(self):
        # Calculate aggregate metrics of the shrinkage vector without tracking gradients
        with torch.no_grad():
            mean_shrinkage = self.learned_shrinkage.mean().item()
            max_shrinkage = self.learned_shrinkage.max().item()
            min_shrinkage = self.learned_shrinkage.min().item()
            
        print(f"\n--- Epoch {self.current_epoch} End ---")
        print(f"Shrinkage Vector -> Mean: {mean_shrinkage:.4f} | Max: {max_shrinkage:.4f} | Min: {min_shrinkage:.4f}")
        
        # Log the mean to the progress bar so you can still track it
        self.log("shrinkage_mean", mean_shrinkage, prog_bar=True)

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