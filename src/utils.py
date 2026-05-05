import os

import numpy as np
import pandas as pd
import torch

def build_correlation_adjacency(df: pd.DataFrame, static_df: pd.DataFrame, val_start_date: str) -> tuple:
    """
    Builds Data-Driven Complement and Cannibalization graphs using Pearson Correlation 
    of historical training sales. Eliminates human categorical bias.
    """
    print("Calculating Data-Driven Pearson Correlation Matrices...")
    
    # 1. Isolate Training Data (Strictly prevent future data leakage!)
    train_df = df[df['ds'] < val_start_date].copy()
    
    # 2. Pivot to get [Time x Nodes] matrix
    # Fill missing days with 0 to ensure accurate variance calculation
    pivot_df = train_df.pivot(index='ds', columns='unique_id', values='y').fillna(0)
    
    # Ensure the columns exactly match the sorted order of your PyTorch Tensors
    node_order = static_df['unique_id'].tolist()
    pivot_df = pivot_df[node_order]
    
    # 3. Calculate Pearson Correlation Matrix [Nodes x Nodes]
    corr_matrix = pivot_df.corr(method='pearson').values
    
    # Fill NaNs (Items with completely flat 0 sales have zero variance, resulting in NaN correlation)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # 4. Split into True Complements and True Substitutes
    # We use thresholds to act as a noise filter. We only want strong economic signals.
    pos_threshold = 0.05  # Positive correlation = Halo Effect
    neg_threshold = -0.05 # Negative correlation = Cannibalization
    
    adj_comp = np.where(corr_matrix > pos_threshold, corr_matrix, 0.0)
    
    # For cannibalization, we take the ABSOLUTE value. The GNN spatial routing requires 
    # positive edge weights to pass the signal. The network's alpha scalars will handle the rest.
    adj_canni = np.where(corr_matrix < neg_threshold, np.abs(corr_matrix), 0.0) 
    
    # 5. The Geographic Firewall (Store Mask)
    # Even if sales correlate, items in TX_1 cannot physically interact with items in TX_2
    # store_vals = static_df['store_id'].values
    # store_mask = (store_vals[:, None] == store_vals[None, :]).astype(float)
    
    # adj_comp = adj_comp * store_mask
    # adj_canni = adj_canni * store_mask
    
    # Remove self-loops from cannibalization (an item cannot substitute itself)
    np.fill_diagonal(adj_canni, 0.0)
    
    # 6. Row-Normalize to prevent exploding gradients during message passing
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