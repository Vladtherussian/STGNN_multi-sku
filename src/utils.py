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