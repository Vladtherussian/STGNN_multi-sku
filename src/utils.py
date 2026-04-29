import os

import numpy as np
import pandas as pd
import torch


def build_weighted_adjacency(static_df: pd.DataFrame, stat_exog_cols: list, volume_col: str = None) -> torch.Tensor:
    """
    Builds a normalized, weighted adjacency matrix based on shared static attributes,
    using hierarchical weighting and a strict department mask for sparsity.
    Optionally weights the edges by the target node's historical volume.
    """
    n_nodes = len(static_df)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    weights = {
        "item_id": 1.0,  
        "dept_id": 0.5,  
        "cat_id": 0.1,   
    }
    
    for col in stat_exog_cols:
        if col in weights:
            vals = static_df[col].values
            match_matrix = (vals[:, None] == vals[None, :]).astype(float)
            adj_matrix += match_matrix * weights[col]
            
    # Hard Sparsity Mask
    dept_vals = static_df['dept_id'].values
    dept_mask = (dept_vals[:, None] == dept_vals[None, :]).astype(float)
    adj_matrix = adj_matrix * dept_mask 

    # --- NEW: GEOGRAPHIC FIREWALL (Store Mask) ---
    # Strictly zero-out any connections between different physical locations
    store_vals = static_df['store_id'].values
    store_mask = (store_vals[:, None] == store_vals[None, :]).astype(float)
    adj_matrix = adj_matrix * store_mask
    
    # --- NEW: VOLUME WEIGHTING ---
    # If volume data is provided, scale the incoming edges by the node's volume
    if volume_col and volume_col in static_df.columns:
        volumes = static_df[volume_col].values
        # Add a tiny constant (1e-5) so items with 0 average sales don't completely zero out their graph
        volumes = np.clip(volumes, a_min=1e-5, a_max=None)
        
        # Multiply each column by the target node's volume
        # This makes connections TO high-volume nodes stronger
        # adj_matrix = adj_matrix * volumes[None, :]
        volumes_norm = volumes / (volumes.max() + 1e-9)
        adj_matrix = adj_matrix * volumes_norm[None, :]

    # Row-normalize so graph convolutions don't explode
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    adj_matrix = adj_matrix / row_sums
    
    return torch.tensor(adj_matrix, dtype=torch.float32)



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

        # MSE
        group_mse = eval_df.groupby('dept_name', as_index=False)['se'].mean().rename(columns={'se': 'value'}).assign(metric='mse', Model=model)

        # category bias: average bias per category (e.g., dept_id)
        group_bias = eval_df.groupby('dept_name', as_index=False).apply(lambda x: (x[model].sum() - x['y'].sum()) / (x['y'].sum() + 1e-9))\
            .rename(columns={None: 'value'}).assign(metric='bias', Model=model)

        grouped_metrics = pd.concat([grouped_metrics, group_wrmsse, group_mae, group_mase, group_mse, group_bias], ignore_index=True)
        item_results = pd.concat([item_results, item_metrics.assign(Model=model)], ignore_index=True)  # Append model name for analysis
        
        # Aggregate global metrics
        wrmsse = (item_metrics['rmsse'] * item_metrics['weight']).sum()
        mae = item_metrics['ae'].mean()
        mse = item_metrics['se'].mean()
        mase = item_metrics['mase'].mean() # Overall MASE is the unweighted average across items
        
        results.append({
            'Model': model,
            'MAE': mae,
            'MSE': mse,
            'MASE': mase,
            'WRMSSE': wrmsse,
            'Bias': bias
        })
    pred_dir = os.path.join(os.getcwd(), "data", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    item_results.to_csv(os.path.join(pred_dir, f"item_metrics_all_models.csv"), index=False)  # Save item-level metrics for analysis
    grouped_metrics.to_csv(os.path.join(pred_dir, f"category_metrics_all_models.csv"), index=False)  # Save category-level metrics for analysis
        
    return pd.DataFrame(results).sort_values('WRMSSE'), grouped_metrics.sort_values(['dept_name', 'Model', 'metric'])