import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv  # used by VanillaSTGNNBlock only
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F

def sparsemax(X, dim=-1):
    """A clean, fast PyTorch implementation of sparsemax for hard cutoff routing."""
    X_sorted, _ = torch.sort(X, dim=dim, descending=True)
    cssv = torch.cumsum(X_sorted, dim=dim) - 1
    ind = torch.arange(1, X.shape[dim] + 1, device=X.device, dtype=X.dtype)
    
    # Add broadcasting dimensions to `ind`
    for _ in range(X.dim() - dim - 1 if dim >= 0 else -dim - 1):
        ind = ind.unsqueeze(-1)
    
    cond = X_sorted - cssv / ind > 0
    rho = cond.sum(dim=dim, keepdim=True)
    tau = torch.gather(cssv, dim=dim, index=rho - 1) / rho
    return torch.clamp(X - tau, min=0.0)

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

    def forward(self, x, adj_comp, adj_canni):
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
        
        # 4. MULTI-GRAPH Spatial Mixing (Pre-Norm)
        x_norm = self.norm_spat(x)
        x_proj = self.spatial_lin(x_norm)
        
        x_proj_2d = x_proj.reshape(B, N, S * Feat)
        
        # Calculate parallel signals for Complements and Cannibalization
        x_spat_comp = torch.bmm(adj_comp, x_proj_2d)
        x_spat_canni = torch.bmm(adj_canni, x_proj_2d)
        
        # Sum the orthogonal spatial signals together
        x_spatial = (x_spat_comp + x_spat_canni).reshape(B, N, S, Feat) + self.spatial_bias
            
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
    def __init__(self, seq_len, pred_len, n_nodes, in_features, hidden_features, n_futr_features, zero_ratio, n_blocks=2, top_k=5, ablation_mode="full", use_zip=False):
        super().__init__()
        self.use_zip = use_zip
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_nodes = n_nodes
        self.top_k = top_k # Restrict the graph to the top 3 strongest connections
        self.ablation_mode = ablation_mode # "full", "static_graph", or "no_graph"
        
        # Trainable identities
        # self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden_features))
        if self.ablation_mode == "full":
            # Register the static Sparsity Prior [N]
            self.register_buffer('zero_ratio', zero_ratio) 

            # The Sparsity Gates: Maps historical sparsity to graph trust
            self.sparsity_gate_comp = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid() 
            )
            self.sparsity_gate_canni = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid() 
            )
        else:
            self.register_buffer("alpha_comp", torch.tensor(1.0))
            self.register_buffer("alpha_canni", torch.tensor(1.0))
        
        
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
        
        # THE FIX: Dual Dynamic Attention Heads
        # Head 1: Learns to dynamically route Halo effects
        self.query_comp = nn.Linear(hidden_features, hidden_features)
        self.key_comp = nn.Linear(hidden_features, hidden_features)
        
        # Head 2: Learns to dynamically route Substitution effects
        self.query_canni = nn.Linear(hidden_features, hidden_features)
        self.key_canni = nn.Linear(hidden_features, hidden_features)
        
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

        # NEW: Time-to-Event Auxiliary Head
        self.tte_head = nn.Sequential(
            nn.Linear(hidden_features, 1),
            nn.ReLU()
        )

        # The final head now accepts (Historical Graph Memory) + (Future Info)
        out_dim = pred_len * 2 if self.use_zip else pred_len
        self.head = nn.Sequential(nn.Linear((seq_len * hidden_features) + (pred_len * hidden_features), out_dim))
        

    def forward(self, x, x_future, adj_comp, adj_canni):
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
            
            # B. Generate Dual Queries and Keys
            Q_c, K_c = self.query_comp(x_context), self.key_comp(x_context)
            Q_k, K_k = self.query_canni(x_context), self.key_canni(x_context)
            
            # C. Compute Dual Dynamic Adjacencies
            dyn_comp = torch.bmm(Q_c, K_c.transpose(1, 2)) / (Q_c.shape[-1] ** 0.5)
            dyn_canni = torch.bmm(Q_k, K_k.transpose(1, 2)) / (Q_k.shape[-1] ** 0.5)
            
            # D. Clamp & ReLU
            dyn_comp = F.relu(dyn_comp.clamp(min=-10.0, max=10.0))
            dyn_canni = F.relu(dyn_canni.clamp(min=-10.0, max=10.0))
            
            # E. Dual Top-K Sparsification
            safe_k = min(self.top_k, N)
            
            # Sparsify Complement Graph
            comp_vals, comp_idx = torch.topk(dyn_comp, k=safe_k, dim=2)
            mask_comp = torch.zeros_like(dyn_comp).scatter_(2, comp_idx, 1.0)
            # THE FIX: Mask with a large negative number before applying sparsemax
            dyn_comp_masked = dyn_comp.masked_fill(mask_comp == 0, -1e4)
            dyn_comp = sparsemax(dyn_comp_masked, dim=2)
            
            # Sparsify Cannibalization Graph
            canni_vals, canni_idx = torch.topk(dyn_canni, k=safe_k, dim=2)
            mask_canni = torch.zeros_like(dyn_canni).scatter_(2, canni_idx, 1.0)
            dyn_canni_masked = dyn_canni.masked_fill(mask_canni == 0, -1e4)
            dyn_canni = sparsemax(dyn_canni_masked, dim=2)

            # F. Dual Blend with static business rules
            static_comp_batch = adj_comp.unsqueeze(0).repeat(B, 1, 1)
            static_canni_batch = adj_canni.unsqueeze(0).repeat(B, 1, 1)

            # Node-level alpha conditioned on sparsity [1, N, 1]
            alpha_c = self.sparsity_gate_comp(self.zero_ratio.unsqueeze(-1)).unsqueeze(0)
            alpha_k = self.sparsity_gate_canni(self.zero_ratio.unsqueeze(-1)).unsqueeze(0)
            
            # Blend the specialized dynamic maps into their respective static graphs
            combined_comp = (alpha_c * static_comp_batch) + ((1 - alpha_c) * dyn_comp)
            combined_canni = (alpha_k * static_canni_batch) + ((1 - alpha_k) * dyn_canni)
            
            combined_comp = F.normalize(combined_comp, p=1, dim=2)
            combined_canni = F.normalize(combined_canni, p=1, dim=2)

        # 4. Message Passing with Multi-Graph Tensors
        from torch.utils.checkpoint import checkpoint
        for block in self.blocks:
            x = checkpoint(block, x, combined_comp, combined_canni, use_reentrant=False)
            
        x_hist_flat = x.reshape(B, N, S * x.shape[-1])
        
        # 5. Process Future Covariates
        x_futr_proj = self.futr_projection(x_future)
        x_futr_flat = x_futr_proj.reshape(B, N, -1)
        
        # 6. Predict (No Gate!)
        combined = torch.cat([x_hist_flat, x_futr_flat], dim=2)
        sales_forecast = self.head(combined)                        
        
        # NEW: Predict TTE from the final spatial-temporal GRU state
        x_final = x[:, :, -1, :] # Shape: [B, N, Feat]
        tte_pred = self.tte_head(x_final).squeeze(-1) # Shape: [B, N]
        
        # Split the output into Probability Logits and Poisson Rate
        if self.use_zip:
            pi_logits = sales_forecast[:, :, :self.pred_len]
            lambda_norm = sales_forecast[:, :, self.pred_len:]
            return pi_logits, lambda_norm, tte_pred
            
        return sales_forecast, tte_pred


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
    def __init__(self, model, adj_comp, adj_canni, learning_rate=1e-3, use_zip=False):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.use_zip = use_zip
        self.register_buffer("adj_comp", adj_comp)
        self.register_buffer("adj_canni", adj_canni)

    def forward(self, x, x_future):
        return self.model(x, x_future, self.adj_comp, self.adj_canni)

    def training_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        y_hist_fp32 = y_hist.float()
        y_fp32 = y.float()
        mean = y_hist_fp32.mean(dim=-1, keepdim=True)
        var = y_hist_fp32.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        y_clamp = y.float().clamp(min=0.0)

        # ==========================================
        # THE ZIP FORWARD PASS & LOSS
        # ==========================================
        if self.use_zip:
            # THE FIX: Unpack the new tte_pred
            pi_logits, lambda_norm, tte_pred = self(x, x_future)
            
            lambda_raw = (lambda_norm.float() * std) + mean
            rate = F.softplus(lambda_raw) + 1e-4
            pi = torch.sigmoid(pi_logits.float())
            
            # ZIP Log Likelihood Formula
            prob_zero = pi + (1.0 - pi) * torch.exp(-rate)
            log_prob_zero = torch.log(prob_zero + 1e-8)
            log_prob_pos = torch.log(1.0 - pi + 1e-8) + y_clamp * torch.log(rate + 1e-8) - rate - torch.lgamma(y_clamp + 1.0)
            
            is_zero = (y_clamp == 0.0).float()
            primary_loss = -(is_zero * log_prob_zero + (1.0 - is_zero) * log_prob_pos).mean()
            y_hat = (1.0 - pi) * rate
            
        else: # Standard Poisson Fallback
            # THE FIX: Unpack the new tte_pred
            y_hat_norm, tte_pred = self(x, x_future)
            y_hat_raw = (y_hat_norm.float() * std) + mean
            y_hat = F.softplus(y_hat_raw) 
            primary_loss = F.poisson_nll_loss(y_hat, y_clamp, log_input=False)

        # Graph Consistency Loss 
        log_y = torch.log1p(y_clamp)
        log_y_hat = torch.log1p(y_hat)
        residuals = log_y - log_y_hat          
        res_mean = residuals.mean(dim=-1)          
        
        neighbor_res = torch.bmm(
            self.adj_comp.float().unsqueeze(0).expand(residuals.shape[0], -1, -1),
            res_mean.unsqueeze(-1)
        ).squeeze(-1)
        graph_loss = F.mse_loss(res_mean, neighbor_res.detach())
        
        # ==========================================
        # NEW: TIME-TO-EVENT (TTE) AUXILIARY LOSS
        # ==========================================
        # Find the exact index of the first sale in the prediction horizon
        has_sale = (y_clamp > 0).any(dim=2)
        first_sale_idx = torch.argmax((y_clamp > 0).float(), dim=2).float()
        
        # If no sale exists in the entire 28-day horizon, the TTE is 28
        true_tte = torch.where(has_sale, first_sale_idx, torch.tensor(y_clamp.shape[2], dtype=torch.float32, device=y_clamp.device))
        
        # Calculate the Huber loss between predicted gap and actual gap
        tte_loss = F.huber_loss(tte_pred, true_tte, delta=1.0)

        # Blend all 3 losses! (TTE gets a small weight so it acts as a regularizer)
        loss = primary_loss + (0.1 * graph_loss) + (0.01 * tte_loss)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('tte_loss', tte_loss, prog_bar=True) # Great for monitoring!
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        # --- FP32 ARMOR ---
        y_hist_fp32 = y_hist.float()
        y_fp32 = y.float()
        
        mean = y_hist_fp32.mean(dim=-1, keepdim=True)
        var = y_hist_fp32.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        y_clamp = y.float().clamp(min=0.0)

        # ==========================================
        # THE ZIP FORWARD PASS & LOSS
        # ==========================================
        if self.use_zip:
            # THE FIX: Unpack the new tte_pred
            pi_logits, lambda_norm, tte_pred = self(x, x_future)
            
            lambda_raw = (lambda_norm.float() * std) + mean
            rate = F.softplus(lambda_raw) + 1e-4
            pi = torch.sigmoid(pi_logits.float())
            
            # ZIP Log Likelihood Formula
            prob_zero = pi + (1.0 - pi) * torch.exp(-rate)
            log_prob_zero = torch.log(prob_zero + 1e-8)
            log_prob_pos = torch.log(1.0 - pi + 1e-8) + y_clamp * torch.log(rate + 1e-8) - rate - torch.lgamma(y_clamp + 1.0)
            
            is_zero = (y_clamp == 0.0).float()
            primary_loss = -(is_zero * log_prob_zero + (1.0 - is_zero) * log_prob_pos).mean()
            y_hat = (1.0 - pi) * rate
            
        else: # Standard Poisson Fallback
            # THE FIX: Unpack the new tte_pred
            y_hat_norm, tte_pred = self(x, x_future)
            y_hat_raw = (y_hat_norm.float() * std) + mean
            y_hat = F.softplus(y_hat_raw) 
            primary_loss = F.poisson_nll_loss(y_hat, y_clamp, log_input=False)

        # Graph Consistency Loss 
        log_y = torch.log1p(y_clamp)
        log_y_hat = torch.log1p(y_hat)
        residuals = log_y - log_y_hat          
        res_mean = residuals.mean(dim=-1)          
        
        neighbor_res = torch.bmm(
            self.adj_comp.float().unsqueeze(0).expand(residuals.shape[0], -1, -1),
            res_mean.unsqueeze(-1)
        ).squeeze(-1)
        graph_loss = F.mse_loss(res_mean, neighbor_res.detach())
        
        # ==========================================
        # NEW: TIME-TO-EVENT (TTE) AUXILIARY LOSS
        # ==========================================
        # Find the exact index of the first sale in the prediction horizon
        has_sale = (y_clamp > 0).any(dim=2)
        first_sale_idx = torch.argmax((y_clamp > 0).float(), dim=2).float()
        
        # If no sale exists in the entire 28-day horizon, the TTE is 28
        true_tte = torch.where(has_sale, first_sale_idx, torch.tensor(y_clamp.shape[2], dtype=torch.float32, device=y_clamp.device))
        
        # Calculate the Huber loss between predicted gap and actual gap
        tte_loss = F.huber_loss(tte_pred, true_tte, delta=1.0)

        # Blend all 3 losses! (TTE gets a small weight so it acts as a regularizer)
        loss = primary_loss + (0.1 * graph_loss) + (0.01 * tte_loss)
        
        self.log('val_loss', loss, prog_bar=True)   # <--- FIXED
        self.log('val_tte_loss', tte_loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, _batch_idx):
        x, y_hist, x_future, y = batch
        
        # --- FP32 ARMOR ---
        y_hist_fp32 = y_hist.float()
        y_fp32 = y.float()
        
        mean = y_hist_fp32.mean(dim=-1, keepdim=True)
        var = y_hist_fp32.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        y_clamp = y.float().clamp(min=0.0)

        # ==========================================
        # THE ZIP FORWARD PASS & LOSS
        # ==========================================
        if self.use_zip:
            # THE FIX: Unpack the new tte_pred
            pi_logits, lambda_norm, tte_pred = self(x, x_future)
            
            lambda_raw = (lambda_norm.float() * std) + mean
            rate = F.softplus(lambda_raw) + 1e-4
            pi = torch.sigmoid(pi_logits.float())
            
            # ZIP Log Likelihood Formula
            prob_zero = pi + (1.0 - pi) * torch.exp(-rate)
            log_prob_zero = torch.log(prob_zero + 1e-8)
            log_prob_pos = torch.log(1.0 - pi + 1e-8) + y_clamp * torch.log(rate + 1e-8) - rate - torch.lgamma(y_clamp + 1.0)
            
            is_zero = (y_clamp == 0.0).float()
            primary_loss = -(is_zero * log_prob_zero + (1.0 - is_zero) * log_prob_pos).mean()
            y_hat = (1.0 - pi) * rate
            
        else: # Standard Poisson Fallback
            # THE FIX: Unpack the new tte_pred
            y_hat_norm, tte_pred = self(x, x_future)
            y_hat_raw = (y_hat_norm.float() * std) + mean
            y_hat = F.softplus(y_hat_raw) 
            primary_loss = F.poisson_nll_loss(y_hat, y_clamp, log_input=False)

        # Graph Consistency Loss 
        log_y = torch.log1p(y_clamp)
        log_y_hat = torch.log1p(y_hat)
        residuals = log_y - log_y_hat          
        res_mean = residuals.mean(dim=-1)          
        
        neighbor_res = torch.bmm(
            self.adj_comp.float().unsqueeze(0).expand(residuals.shape[0], -1, -1),
            res_mean.unsqueeze(-1)
        ).squeeze(-1)
        graph_loss = F.mse_loss(res_mean, neighbor_res.detach())
        
        # ==========================================
        # NEW: TIME-TO-EVENT (TTE) AUXILIARY LOSS
        # ==========================================
        # Find the exact index of the first sale in the prediction horizon
        has_sale = (y_clamp > 0).any(dim=2)
        first_sale_idx = torch.argmax((y_clamp > 0).float(), dim=2).float()
        
        # If no sale exists in the entire 28-day horizon, the TTE is 28
        true_tte = torch.where(has_sale, first_sale_idx, torch.tensor(y_clamp.shape[2], dtype=torch.float32, device=y_clamp.device))
        
        # Calculate the Huber loss between predicted gap and actual gap
        tte_loss = F.huber_loss(tte_pred, true_tte, delta=1.0)

        # Blend all 3 losses! (TTE gets a small weight so it acts as a regularizer)
        loss = primary_loss + (0.1 * graph_loss) + (0.01 * tte_loss)
        
        self.log('test_loss', loss, prog_bar=True)   # <--- FIXED
        self.log('test_tte_loss', tte_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_train_epoch_end(self):
        # Safely check if we are using the MLP gates
        if hasattr(self.model, 'sparsity_gate_comp'):
            with torch.no_grad():
                # Push the static zero_ratio through the MLPs to get the current alpha vectors
                current_alpha_c = self.model.sparsity_gate_comp(self.model.zero_ratio.unsqueeze(-1))
                current_alpha_k = self.model.sparsity_gate_canni(self.model.zero_ratio.unsqueeze(-1))
                
                # Calculate metrics directly (the MLPs already output Sigmoid [0,1], so no need to apply it again!)
                alpha_c_mean = current_alpha_c.mean().item()
                alpha_k_mean = current_alpha_k.mean().item()
                
                alpha_c_std = current_alpha_c.std().item()
                alpha_k_std = current_alpha_k.std().item()
                
            print(f"\n--- Epoch {self.current_epoch} End ---")
            print(f"Alpha Complement (Mean): {alpha_c_mean:.4f} (Std: {alpha_c_std:.4f}) | Alpha Canni (Mean): {alpha_k_mean:.4f} (Std: {alpha_k_std:.4f})")
            self.log("alpha_comp", alpha_c_mean, prog_bar=True)
            self.log("alpha_canni", alpha_k_mean, prog_bar=True)


class LitResidualSTGNN(pl.LightningModule):
    """
    Dedicated Lightning wrapper for the Residual Two-Stage Model. 
    Now features a self-learning shrinkage parameter to automatically 
    calibrate the magnitude of the spatial corrections!
    """
    def __init__(self, model, adj_comp, adj_canni, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        
        # THE FIX: Register the dual graphs instead of a single matrix
        self.register_buffer("adj_comp", adj_comp)
        self.register_buffer("adj_canni", adj_canni)
        
        self.learned_shrinkage = nn.Parameter(torch.ones(1, model.n_nodes, 1) * 0.1)

    def forward(self, pure_x, x_future):
        # THE FIX: Unpack the tuple (ignore the TTE prediction with an underscore)
        raw_residual, _ = self.model(pure_x, x_future, self.adj_comp, self.adj_canni)
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