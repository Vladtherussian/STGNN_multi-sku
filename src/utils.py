import numpy as np
import pandas as pd
import torch

def build_weighted_adjacency(static_df: pd.DataFrame, stat_exog_cols: list) -> torch.Tensor:
    """
    Builds a normalized, weighted adjacency matrix based on shared static attributes.
    """

    # static_df must be sorted in the exact same order as your unique_ids in the tensor
    n_nodes = len(static_df)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    # Iterate through every static category (store, item, dept, etc.)
    for col in stat_exog_cols:
        vals = static_df[col].values
        
        # Vectorized operation: Creates an N x N boolean matrix where [i, j] is True 
        # if node i and node j share the exact same value for this category.
        match_matrix = (vals[:, None] == vals[None, :]).astype(float)
        
        # Add it to the total adjacency matrix
        adj_matrix += match_matrix
        
    # Optional but recommended: Graph Convolutions require normalized weights so 
    # the feature values don't explode during matrix multiplication.
    # We row-normalize the matrix so the sum of all connections for a single node equals 1.0.
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    
    # Prevent division by zero just in case
    row_sums[row_sums == 0] = 1.0 
    adj_matrix = adj_matrix / row_sums
    
    return torch.tensor(adj_matrix, dtype=torch.float32)


def calculate_m5_metrics(train_df, test_df, predictions_df, models, pred_len=28):
    """
    Calculates MAE, Bias, and a localized WRMSSE for a suite of models.
    
    train_df: The historical training data (used to calculate scaling factors and weights)
    test_df: The actual true sales for the 28-day holdout
    predictions_df: The dataframe containing columns for each model's forecast
    models: List of string names of the models (e.g., ['TSMixerx', 'TFT', 'AutoARIMA'])
    """
    results = []
    
    # 1. Calculate the Scaling Factor (Denominator of RMSSE)
    # This is the mean squared day-to-day difference in the training set for each item
    # We drop NAs to ignore days before the item was introduced to the shelf
    diff_sq = train_df.groupby('unique_id')['y'].apply(lambda x: (x.diff() ** 2).mean()).reset_index()
    diff_sq.rename(columns={'y': 'scale'}, inplace=True)
    
    # 2. Calculate the Revenue Weights (Numerator of WRMSSE)
    # Get the last pred_len days of the training set to calculate recent revenue
    last_train = train_df.groupby('unique_id').tail(pred_len).copy()
    last_train['revenue'] = last_train['y'] * last_train['price']
    
    weights = last_train.groupby('unique_id')['revenue'].sum().reset_index()
    total_revenue = weights['revenue'].sum()
    weights['weight'] = weights['revenue'] / total_revenue
    
    # Merge test data with predictions
    eval_df = test_df[['unique_id', 'ds', 'y']].merge(predictions_df, on=['unique_id', 'ds'], how='inner')
    
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
        
        # Merge scaling factor and weights
        item_metrics = item_errors.merge(diff_sq, on='unique_id').merge(weights, on='unique_id')
        
        # Calculate RMSSE per item
        # If scale is 0 (item never sold or sold exactly the same amount every day), prevent division by zero
        item_metrics['rmsse'] = np.sqrt(item_metrics['se'] / (item_metrics['scale'] + 1e-9))
        
        # Calculate WRMSSE
        wrmsse = (item_metrics['rmsse'] * item_metrics['weight']).sum()
        mae = item_metrics['ae'].mean()
        
        results.append({
            'Model': model,
            'MAE': mae,
            'WRMSSE': wrmsse,
            'Bias': bias
        })
        
    return pd.DataFrame(results).sort_values('WRMSSE')