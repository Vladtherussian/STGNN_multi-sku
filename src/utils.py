import os
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats

def fast_gpu_granger(Y_series, X_series, max_lag=1, chunk_size=10000):
    """
    Batched GPU Granger Causality Test.
    Y_series: [Num_Pairs, Time] (The target items)
    X_series: [Num_Pairs, Time] (The predictor items)
    Returns: p_values [Num_Pairs] array
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N_pairs, T = Y_series.shape
    p_values_all = np.ones(N_pairs)
    
    # Process in chunks to respect VRAM limits
    for i in range(0, N_pairs, chunk_size):
        end_idx = min(i + chunk_size, N_pairs)
        
        # Move chunk to GPU
        Y_chunk = torch.tensor(Y_series[i:end_idx], dtype=torch.float32, device=device)
        X_chunk = torch.tensor(X_series[i:end_idx], dtype=torch.float32, device=device)
        
        # Create Lagged Tensors: Shape [Batch, Time-1, 1]
        Y_target = Y_chunk[:, 1:].unsqueeze(-1)
        Y_lagged = Y_chunk[:, :-1].unsqueeze(-1)
        X_lagged = X_chunk[:, :-1].unsqueeze(-1)
        ones = torch.ones_like(Y_lagged)
        
        # Build the Design Matrices
        A_R = torch.cat([Y_lagged, ones], dim=-1)                # Restricted
        A_UR = torch.cat([Y_lagged, X_lagged, ones], dim=-1)     # Unrestricted
        
        # Solve Least Squares (Batched)
        beta_R = torch.linalg.lstsq(A_R, Y_target).solution
        beta_UR = torch.linalg.lstsq(A_UR, Y_target).solution
        
        # Calculate Residuals via Batched Matrix Multiplication
        pred_R = torch.bmm(A_R, beta_R)
        pred_UR = torch.bmm(A_UR, beta_UR)
        
        # Sum of Squared Residuals [Batch]
        SSR_R = torch.sum((Y_target - pred_R)**2, dim=[1, 2]).cpu().numpy()
        SSR_UR = torch.sum((Y_target - pred_UR)**2, dim=[1, 2]).cpu().numpy()
        
        # Calculate F-Statistic
        df_num = 1
        df_denom = Y_target.shape[1] - A_UR.shape[2]
        
        # Safeguard against zero variance division
        SSR_UR = np.where(SSR_UR == 0, 1e-10, SSR_UR)
        
        f_stats = ((SSR_R - SSR_UR) / df_num) / (SSR_UR / df_denom)
        f_stats = np.clip(f_stats, 0, None) # Clip numerical precision errors
        
        # Calculate P-Values on CPU using SciPy
        p_vals = stats.f.sf(f_stats, df_num, df_denom)
        p_values_all[i:end_idx] = p_vals
        
    return p_values_all


def build_correlation_adjacency(df: pd.DataFrame, static_df: pd.DataFrame, val_start_date: str, granger: bool = False) -> tuple:
    """
    Builds Asymmetric, Directed Causal Graphs using Pre-Filtered GPU Granger Causality,
    or falls back to undirected Pearson if granger=False.
    """
    print(f"Building {'Granger' if granger else 'Pearson'} Causality Matrices...")
    
    # 1. Isolate Training Data 
    train_df = df[df['ds'] < val_start_date].copy()
    
    # 2. Pivot to get [Time x Nodes] matrix
    pivot_df = train_df.pivot(index='ds', columns='unique_id', values='y').fillna(0)
    
    # Ensure columns match PyTorch Tensor indices
    node_order = static_df['unique_id'].tolist()
    pivot_df = pivot_df[node_order]
    
    # 3. Differenced Data (Granger Causality requires Stationary Data)
    diff_df = pivot_df.diff().dropna()
    diff_values = diff_df.values.T # Transpose to [Nodes, Time] for fast indexing
    
    # 4. Fast Undirected Pass (Pearson) to find Candidates
    corr_matrix = diff_df.corr(method='pearson').values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    pos_threshold = 0.05
    neg_threshold = -0.05
    
    # Get indices of candidate pairs (y_idx = Target, x_idx = Predictor)
    comp_y_idx, comp_x_idx = np.where(corr_matrix > pos_threshold)
    canni_y_idx, canni_x_idx = np.where(corr_matrix < neg_threshold)
    
    # Remove self-loops
    comp_mask = comp_y_idx != comp_x_idx
    comp_y_idx, comp_x_idx = comp_y_idx[comp_mask], comp_x_idx[comp_mask]
    
    canni_mask = canni_y_idx != canni_x_idx
    canni_y_idx, canni_x_idx = canni_y_idx[canni_mask], canni_x_idx[canni_mask]
    
    # THE FIX: Initialize the matrices outside the if-statement!
    N = len(node_order)
    adj_comp = np.zeros((N, N), dtype=np.float32)
    adj_canni = np.zeros((N, N), dtype=np.float32)

    if granger:
        print(f"Pre-filter found {len(comp_y_idx)} Complement candidates and {len(canni_y_idx)} Cannibalization candidates.")
        print("Running GPU Batched Granger Causal Discovery...")
        
        # ---------------- GRANGER COMPLEMENTS ----------------
        if len(comp_y_idx) > 0:
            Y_comp = diff_values[comp_y_idx]
            X_comp = diff_values[comp_x_idx]
            p_vals_comp = fast_gpu_granger(Y_comp, X_comp)
            
            # Keep only statistically significant causal edges (p < 0.05)
            sig_comp = p_vals_comp < 0.05
            weights_comp = corr_matrix[comp_y_idx[sig_comp], comp_x_idx[sig_comp]]
            adj_comp[comp_y_idx[sig_comp], comp_x_idx[sig_comp]] = weights_comp
            
        # ---------------- GRANGER CANNIBALIZATION ----------------
        if len(canni_y_idx) > 0:
            Y_canni = diff_values[canni_y_idx]
            X_canni = diff_values[canni_x_idx]
            p_vals_canni = fast_gpu_granger(Y_canni, X_canni)
            
            # Keep only statistically significant causal edges (p < 0.05)
            sig_canni = p_vals_canni < 0.05
            weights_canni = np.abs(corr_matrix[canni_y_idx[sig_canni], canni_x_idx[sig_canni]])
            adj_canni[canni_y_idx[sig_canni], canni_x_idx[sig_canni]] = weights_canni
            
    else:
        # THE FIX: Fallback to standard Pearson if granger=False
        if len(comp_y_idx) > 0:
            adj_comp[comp_y_idx, comp_x_idx] = corr_matrix[comp_y_idx, comp_x_idx]
        if len(canni_y_idx) > 0:
            adj_canni[canni_y_idx, canni_x_idx] = np.abs(corr_matrix[canni_y_idx, canni_x_idx])

    # 6. Row-Normalize to prevent gradient explosion
    def normalize_graph(matrix):
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        return matrix / row_sums
        
    adj_comp = normalize_graph(adj_comp)
    adj_canni = normalize_graph(adj_canni)
    
    print(f"Graph Density - Complements: {(adj_comp > 0).mean():.4%} | Cannibalization: {(adj_canni > 0).mean():.4%}")
    
    return torch.tensor(adj_comp, dtype=torch.float32), torch.tensor(adj_canni, dtype=torch.float32)



def calculate_m5_metrics(train_df, test_df, predictions_df, models, raw_feature_sales, pred_len=28):
    """
    Calculates MAE, MSE, MASE, Bias, and a localized WRMSSE for a suite of models.
    
    train_df: The historical training data (used to calculate scaling factors and weights)
    test_df: The actual true sales for the 28-day holdout
    predictions_df: The dataframe containing columns for each model's forecast
    models: List of string names of the models (e.g., ['TSMixerx', 'TFT', 'AutoARIMA'])
    """
    raw_prices = raw_feature_sales[['id', 'timestamp', 'sell_price']].copy()
    raw_prices.rename(columns={'id': 'unique_id', 'timestamp': 'ds'}, inplace=True)
    train_df = train_df.merge(raw_prices, on=['unique_id', 'ds'], how='left')
    
    last_train = train_df.groupby('unique_id').tail(pred_len).copy()
    
    # Merge raw price on BOTH unique_id AND ds to avoid many-to-many join
    last_train['revenue'] = last_train['y'] * last_train['sell_price']

    results = []
    
    # 1. Calculate the Scaling Factors (Denominators)
    # RMSSE Denominator: Mean squared day-to-day difference
    diff_sq = train_df.groupby('unique_id')['y'].apply(lambda x: (x.diff() ** 2).mean()).reset_index()
    diff_sq.rename(columns={'y': 'scale_rmsse'}, inplace=True)
    
    # MASE Denominator: Mean absolute day-to-day difference
    diff_abs = train_df.groupby('unique_id')['y'].apply(lambda x: x.diff().abs().mean()).reset_index()
    diff_abs.rename(columns={'y': 'scale_mase'}, inplace=True)

    # Merge scales together
    scales = diff_sq.merge(diff_abs, on='unique_id')
    
    # 2. Calculate the Revenue Weights (Numerator of WRMSSE)
    # Get the last pred_len days of the training set to calculate recent revenue
    last_train = train_df.groupby('unique_id').tail(pred_len).copy()
    # last_train = last_train.merge(raw_feature_sales[['id', 'timestamp', 'sell_price']], left_on=['unique_id', 'ds'], right_on=['id', 'timestamp'], how='left')
    last_train['revenue'] = last_train['y'] * last_train['sell_price']
    
    weights = last_train.groupby('unique_id')['revenue'].sum().reset_index()
    total_revenue = weights['revenue'].sum()
    weights['weight'] = weights['revenue'] / total_revenue
    
    # Merge test data with predictions
    eval_df = test_df[['unique_id', 'ds', 'y']].merge(predictions_df, on=['unique_id', 'ds'], how='inner')
    item_results = pd.DataFrame()  # To store item-level metrics for analysis
    grouped_metrics = pd.DataFrame()  # To store category-level metrics for analysis

    for model in models:
        # Calculate Absolute Error and Squared Error per row
        eval_df['ae'] = (eval_df['y'] - eval_df[model]).abs()
        eval_df['se'] = (eval_df['y'] - eval_df[model]) ** 2
        
        # Calculate Bias (Sum of Forecasts - Sum of Actuals) / Sum of Actuals
        # Positive bias = over-forecasting, Negative bias = under-forecasting
        total_actual = eval_df['y'].sum()
        total_forecast = eval_df[model].sum()
        bias = (total_forecast - total_actual) / (total_actual + 1e-9)
        
        # Aggregate errors by item
        item_errors = eval_df.groupby('unique_id').agg({'ae': 'mean', 'se': 'mean'}).reset_index()
        
        # Merge scaling factors and weights
        item_metrics = item_errors.merge(scales, on='unique_id').merge(weights, on='unique_id')
        
        # Calculate RMSSE per item
        # If scale is 0, prevent division by zero
        item_metrics['rmsse'] = np.sqrt(item_metrics['se'] / (item_metrics['scale_rmsse'] + 1e-9))
        
        # Calculate MASE per item
        item_metrics['mase'] = item_metrics['ae'] / (item_metrics['scale_mase'] + 1e-9)

        #======================
        # Category Calculation
        #======================
        eval_df['dept_name'] = eval_df['unique_id'].str.split("_", expand=True)[0]
        # wrmsse per category        
        item_metrics['dept_name'] = item_metrics['unique_id'].str.split("_", expand=True)[0]
        item_metrics['weighted_rmsse'] = item_metrics['rmsse'] * item_metrics['weight']
        group_wrmsse = item_metrics.groupby('dept_name')['weighted_rmsse'].sum().reset_index()\
            .rename(columns={'weighted_rmsse': 'value'}).assign(metric='weighted_rmsse', Model=model)

        # MAE
        group_mae = eval_df.groupby('dept_name', as_index=False)['ae'].mean().rename(columns={'ae': 'value'}).assign(metric='mae', Model=model)

        # MASE
        group_mase = item_metrics.groupby('dept_name', as_index=False)['mase'].mean().rename(columns={'mase': 'value'}).assign(metric='mase', Model=model)
        
        # RMSE
        group_rmsse = item_metrics.groupby('dept_name', as_index=False)['rmsse'].mean().rename(columns={'rmsse': 'value'}).assign(metric='rmsse', Model=model)

        # MSE
        group_mse = eval_df.groupby('dept_name', as_index=False)['se'].mean().rename(columns={'se': 'value'}).assign(metric='mse', Model=model)

        # category bias: average bias per category (e.g., dept_id)
        group_bias = eval_df.groupby('dept_name', as_index=False).apply(lambda x: (x[model].sum() - x['y'].sum()) / (x['y'].sum() + 1e-9))\
            .rename(columns={None: 'value'}).assign(metric='bias', Model=model)

        grouped_metrics = pd.concat([grouped_metrics, group_wrmsse, group_rmsse, group_mae, group_mase, group_mse, group_bias], ignore_index=True)
        item_results = pd.concat([item_results, item_metrics.assign(Model=model)], ignore_index=True)  # Append model name for analysis
        
        # Aggregate global metrics
        wrmsse = (item_metrics['rmsse'] * item_metrics['weight']).sum()
        mae = item_metrics['ae'].mean()
        mse = item_metrics['se'].mean()
        mase = item_metrics['mase'].mean() # Overall MASE is the unweighted average across items
        rmsse = item_metrics['rmsse'].mean()
        
        results.append({
            'Model': model,
            'MAE': mae,
            'MSE': mse,
            'MASE': mase,
            'WRMSSE': wrmsse,
            'RMSSE': rmsse,
            'Bias': bias
        })
    pred_dir = os.path.join(os.getcwd(), "data", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    item_results.to_csv(os.path.join(pred_dir, f"item_metrics_all_models.csv"), index=False)  # Save item-level metrics for analysis
    grouped_metrics.to_csv(os.path.join(pred_dir, f"category_metrics_all_models.csv"), index=False)  # Save category-level metrics for analysis
        
    return pd.DataFrame(results).sort_values('WRMSSE'), grouped_metrics.sort_values(['dept_name', 'Model', 'metric'])