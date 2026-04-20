import os
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List
import pandas as pd
import gdown
import torch
import numpy as np

from dagster import asset, Config



from pathlib import Path

# __file__ points to assets.py. 
# .parent goes up to 'src'
# .parent.parent goes up to 'STGNN_multi-sku' (Project Root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from .utils import build_weighted_adjacency

# 1. Define a Config class for your parameters
class SalesDataConfig(Config):
    downsample_dataset: bool = True

@asset
def download_m5_data() -> str:
    """Downloads and extracts the M5 dataset, returning the folder path."""
    work_directory = DATA_DIR
    os.makedirs(work_directory, exist_ok=True)

    raw_data_zip = os.path.join(work_directory, "raw.zip")
    
    # Download using gdown to bypass the virus scan warning
    if not os.path.exists(raw_data_zip):
        file_id = "1NYHXmgrcXg50zR4CVWPPntHx9vvU5jbM"
        gdown.download(id=file_id, output=raw_data_zip, quiet=False)

    raw_data_dir = os.path.join(work_directory, "raw")
    
    # Extract the M5 dataset
    if not os.path.exists(raw_data_dir):
        with zipfile.ZipFile(raw_data_zip, "r") as zip_ref:
            zip_ref.extractall(raw_data_dir)
            
    return raw_data_dir

@asset(deps=["download_m5_data"])
def load_sales_data(download_m5_data: str, config: SalesDataConfig) -> tuple:
    """Loads the raw pricing data into memory."""
    # download_m5_data automatically contains the 'raw_data_dir' string
    prices_path = os.path.join(download_m5_data, "sell_prices.csv")
    calendar_path = os.path.join(download_m5_data, "calendar.csv")
    # sales_train_path = os.path.join(download_m5_data, "sales_train_validation.csv")
    sales_train_eval_path = os.path.join(download_m5_data, "sales_train_evaluation.csv")

    prices_df= pd.read_csv(prices_path)
    calendar_df= pd.read_csv(calendar_path)
    # sales_train_df= pd.read_csv(sales_train_path)
    sales_train_eval_df= pd.read_csv(sales_train_eval_path)

    if config.downsample_dataset:
        # Find the top 200 highest volume items dynamically
        sales_col = sales_train_eval_df.columns[sales_train_eval_df.columns.str.startswith("d_")] # Identify the sales columns
        sales_train_eval_df["total_sales"] = sales_train_eval_df[sales_col].sum(axis=1) # Sum across all days to get total sales per item
        sales_train_eval_df = sales_train_eval_df.sort_values("total_sales", ascending=False) # Sort by total sales
        sales_train_eval_df = sales_train_eval_df.head(200) # Keep only the top 200 items
        sales_train_eval_df.drop(columns=["total_sales"], inplace=True) # Drop the helper column

        print("Number of selected items in evaluation set:", len(sales_train_eval_df))

    return prices_df, calendar_df, sales_train_eval_df

@asset(deps=["load_sales_data"])
def process_raw_data(load_sales_data: tuple) -> None:
    """Processes the raw data into a single dataframe."""
    prices_df, calendar_df, sales_train_eval_df = load_sales_data

    #----------------------------------
    # Process Calendar Data
    #----------------------------------
    cleaned_calendar = calendar_df.copy()

    # Temporian and TensorFlow Decision Forests (after) treat NaN values as "missing".
    # In this dataset, a NaN means that there is not calendar event on this day.
    cleaned_calendar.fillna("no_event", inplace=True)

    cleaned_calendar["timestamp"] = cleaned_calendar["date"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d")
    )

    # We keep the mapping wm_yr_wk -> timestamp to clean "raw_sell_prices" in the next section.
    col_to_keep = [
        "weekday", "month", "timestamp", "wm_yr_wk", 
        "event_name_1", "event_type_1", "event_name_2", "event_type_2", 
        "snap_CA", "snap_TX", "snap_WI"
    ]
    wm_yr_wk_map = cleaned_calendar[col_to_keep]
    
    # Set as categorical to reduce memory usage
    for i in ["weekday", "month", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        wm_yr_wk_map[i] = wm_yr_wk_map[i].astype("category")
    

    #----------------------------------
    # Process Sell Price Data
    #----------------------------------
    cleaned_sell_prices = prices_df.copy()
    cleaned_sell_prices = cleaned_sell_prices.merge(wm_yr_wk_map[['timestamp', 'wm_yr_wk']], on="wm_yr_wk", how="left")
    del cleaned_sell_prices["wm_yr_wk"]

    #----------------------------------
    # Process Sales Data
    #----------------------------------
    cleaned_sales = pd.melt(
        sales_train_eval_df,
        var_name="day",
        value_name="sales",
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )
    # Convert to date
    # cleaned_sales["day"] = cleaned_sales["day"].apply(lambda x: int(x[2:]))
    cleaned_sales["day"] = cleaned_sales["day"].str.slice(2).astype(int)
    origin_date = datetime(2011, 1, 29)
    cleaned_sales["timestamp"] = origin_date + pd.to_timedelta(cleaned_sales["day"] - 1, unit="D")

    # del cleaned_sales["id"]

    raw_feature_sales = cleaned_sales\
        .merge(cleaned_sell_prices, on=["item_id", "timestamp", "store_id"], how="left")\
        .merge(wm_yr_wk_map, on="timestamp", how="left")

    # Define path to dump processed data and create directory if it doesn't exist
    processed_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    return raw_feature_sales.to_parquet(os.path.join(processed_dir, "raw_feature_sales.parquet"), index=False)
    

@asset(deps=["process_raw_data"])
def feature_engineering() -> pd.DataFrame:
    # List to track all the features we create, so we can easily reference them in the next asset when we train our model.
    feature_list = []

    raw_feature_sales = pd.read_parquet(os.path.join(DATA_DIR, "processed", "raw_feature_sales.parquet"))
    # Sort by timestamp to ensure correct order for lag features
    raw_feature_sales = raw_feature_sales.sort_values(by=["id", "item_id", "store_id", "timestamp"])

    #----------------------------------
    # Lag feature for sales
    #----------------------------------
    for lag_horizon in [1, 2, 3, 7, 14]:
        feature_name = f"sales_lag_{lag_horizon}"
        raw_feature_sales[feature_name] = raw_feature_sales.groupby(["id", "item_id", "store_id"])["sales"].shift(lag_horizon)
        feature_list.append(feature_name)

    #----------------------------------
    # Rolling window features
    #----------------------------------
    for win_day in [7, 14, 28, 84]:
        # Create the grouped rolling object ONCE per window size and avoid data leakage
        rolling_obj = raw_feature_sales.groupby(["id", "item_id", "store_id"])["sales_lag_1"].rolling(window=win_day)

        # Calculate the native C-optimized metrics. 
        # groupby.rolling() creates a MultiIndex, so we drop the grouping levels to map it cleanly back to the original rows.
        raw_feature_sales[f"sales_mean_{win_day}"] = rolling_obj.mean().reset_index(level=[0, 1, 2], drop=True)
        raw_feature_sales[f"sales_std_{win_day}"] = rolling_obj.std().reset_index(level=[0, 1, 2], drop=True)
        raw_feature_sales[f"sales_roll_sum_{win_day}"] = rolling_obj.sum().reset_index(level=[0, 1, 2], drop=True)

        # Add to your tracking list
        feature_list.extend([
            f"sales_mean_{win_day}", 
            f"sales_std_{win_day}", 
            f"sales_roll_sum_{win_day}"
        ])

    #----------------------------------
    # Encode calendar events as features
    #----------------------------------
    raw_feature_sales["is_weekend"] = raw_feature_sales["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    feature_list.append("is_weekend")
    raw_feature_sales["month_name"] = raw_feature_sales["timestamp"].dt.month_name()
    feature_list.append("month_name")

    raw_feature_sales["snap_CA"] = raw_feature_sales["snap_CA"].astype(int)
    feature_list.append("snap_CA")
    raw_feature_sales["snap_TX"] = raw_feature_sales["snap_TX"].astype(int)
    feature_list.append("snap_TX")
    raw_feature_sales["snap_WI"] = raw_feature_sales["snap_WI"].astype(int)
    feature_list.append("snap_WI")
    raw_feature_sales["is_event"] = ((raw_feature_sales["event_name_1"] != "no_event") 
                                     | (raw_feature_sales["event_name_2"] != "no_event")).astype(int)
    feature_list.append("is_event")

    # Onehot of event type
    # event_type_cols = ["event_type_1", "event_type_2"]
    # raw_feature_sales = pd.get_dummies(raw_feature_sales, columns=event_type_cols, prefix=event_type_cols)
    # feature_list.extend([col for col in raw_feature_sales.columns if col.startswith("event_type_")])

    # Days until next event
    #----------------------------------
    # Create a temporary column that ONLY contains the timestamp if an event is happening today
    raw_feature_sales["next_event_date"] = raw_feature_sales["timestamp"].where(raw_feature_sales["is_event"] == 1)

    # Backward fill the empty rows. This takes the next available event date and drags it UP the column to fill the past dates.
    raw_feature_sales["next_event_date"] = raw_feature_sales.groupby(["id", "item_id", "store_id"])["next_event_date"].bfill()

    # Subtract the current row's date from the upcoming event date, and extract just the days
    raw_feature_sales["days_until_next_event"] = (raw_feature_sales["next_event_date"] - raw_feature_sales["timestamp"]).dt.days

    # If there are no future events at the very end of the dataset, it will leave NaNs. Fill them with 0 and drop the temp column.
    raw_feature_sales["days_until_next_event"] = raw_feature_sales["days_until_next_event"].fillna(0).astype(int)
    raw_feature_sales = raw_feature_sales.drop(columns=["next_event_date"])
    feature_list.append("days_until_next_event")

    #----------------------------------
    # Encode price features
    #----------------------------------
    # We keep the original price as a feature, but we could also create lag features for price, or compute price changes compared to previous days.
    # Compute promo depth
    raw_feature_sales["price"] = raw_feature_sales["sell_price"]
    feature_list.append("price")
    raw_feature_sales['base_price'] = raw_feature_sales.groupby(['id', 'item_id', 'store_id'])['sell_price'].transform(
        lambda x: x.rolling(window=8, min_periods=1).max()
    )
    # Derive the promo depth: (Base Price - Current Price) / Base Price
    raw_feature_sales['promo_depth'] = (raw_feature_sales['base_price'] - raw_feature_sales['sell_price']) / raw_feature_sales['base_price']
    
    # Fill any NaNs (from the first few weeks with no rolling history) with 0
    raw_feature_sales['promo_depth'] = raw_feature_sales['promo_depth'].fillna(0)
    feature_list.append("promo_depth")

    
    #----------------------------------
    # Sales per department
    #----------------------------------
    # TODO: create sales by department feature


    raw_feature_sales = raw_feature_sales.rename(columns={"id": "unique_id"})

    # Export
    #----------------------------------
    
    # Define path to dump processed data and create directory if it doesn't exist
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    raw_feature_sales.to_parquet(os.path.join(processed_dir, "model_input.parquet"), index=False)

    # save our feature list for easy reference in the next asset when we train our model
    feature_list_path = os.path.join(processed_dir, "feature_list.txt")
    with open(feature_list_path, "w") as f:
        for feature in feature_list:
            f.write(feature + "\n")
    
    return raw_feature_sales

    
@asset(deps=["feature_engineering"])
def prepare_ml_data() -> None:
    """Formats, encodes, and scales the data for LightGBM, N-HiTS, and StemGNN."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    #TODO: May need a variable to indicate start of a product on the shelf, so the model doesn't get confused by long periods of 0 sales followed by a sudden spike when the item launches.
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    df = pd.read_parquet(os.path.join(processed_dir, "model_input.parquet"))

    cols_to_drop = [
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
        'sell_price', 'base_price', 'day'
    ]
    df.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')

    #----------------------------------
    # 1. Nixtla Standard Formatting
    #----------------------------------
    # Nixtla libraries require time series to be defined by 'unique_id', 'ds', and 'y'
    # df["unique_id"] = df["item_id"] + "_" + df["store_id"]
    df = df.rename(columns={"timestamp": "ds", "sales": "y"})

    # ---------------------------------------------------------
    # 1. Backfill Strategy: Ensure Rectangular Data
    # ---------------------------------------------------------
    # Create a grid of every unique_id and every date to find gaps
    all_ids = df['unique_id'].unique()
    all_dates = df['ds'].unique()
    grid = pd.MultiIndex.from_product([all_ids, all_dates], names=['unique_id', 'ds']).to_frame(index=False)
    
    # Merge the original data into this grid
    df = pd.merge(grid, df, on=['unique_id', 'ds'], how='left')
    
    # Fill missing target values with 0 (assuming item wasn't on sale yet)
    df['y'] = df['y'].fillna(0)
    
    # Forward fill static metadata (item_id, store_id, etc.) 
    # and then backfill to cover the start of the series
    static_metadata = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df[static_metadata] = df.groupby('unique_id')[static_metadata].ffill().bfill()

    #----------------------------------
    # 2. Categorical Label Encoding
    #----------------------------------
    # Deep learning models cannot process strings. We convert them to integer IDs.
    categorical_cols = static_metadata + ["month_name", "weekday"]
    le = LabelEncoder()
    for col in categorical_cols:
        # Fill NaN categories before encoding
        df[col] = df[col].astype(object).fillna("unknown").astype(str)
        df[col] = le.fit_transform(df[col])

    #----------------------------------
    # 3. Feature Scaling
    #----------------------------------
    # We explicitly select only continuous features. We do NOT want to scale binary flags 
    # like 'is_weekend' or 'snap_CA', and we leave the target 'y' unscaled so Nixtla can handle it.
    continuous_features = ['price', 'promo_depth', 'days_until_next_event']
    continuous_features += [col for col in df.columns if any(x in col for x in ['lag', 'mean', 'std', 'roll_sum'])]

    df[continuous_features] = df[continuous_features].replace([np.inf, -np.inf], np.nan)
    
    # Fill feature NaNs with 0 before scaling to keep the matrix intact
    df[continuous_features] = df[continuous_features].fillna(0)

    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    if np.isnan(df[continuous_features].values).any() or np.isinf(df[continuous_features].values).any():
        raise ValueError("CRITICAL: df contains NaNs or Infs! Check your upstream scaling.")

    # Save the final ML-ready dataset!
    df.to_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"), index=False)

@asset(deps=["prepare_ml_data"])
def prepare_stgnn_tensors() -> None:
    """Reshapes the flat ML data into 3D tensors and builds the multi-relational adjacency matrix."""
    
    # 1. Load the flat ML-ready data
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    
    # Ensure perfect sorting: Nodes first, then Time. 
    df = df.sort_values(by=["unique_id", "ds"])
    
    # 2. Extract the Static DataFrame for the Adjacency Matrix
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"] 
    static_df = df[["unique_id"] + stat_exog].drop_duplicates().sort_values("unique_id")
    
    # 3. Call your custom function to build the graph!
    adj_tensor = build_weighted_adjacency(static_df=static_df, stat_exog_cols=stat_exog)
    
    # 4. Reshape Temporal Features into 3D Tensors
    n_nodes = df["unique_id"].nunique()
    n_timesteps = df["ds"].nunique()
    
    df_numeric = df.select_dtypes(include=[np.number])
    
    ignore_cols = ["unique_id", "ds", "y"] + stat_exog
    temporal_features = [col for col in df_numeric.columns if col not in ignore_cols]
    n_features = len(temporal_features)
    
    # --- THE SAFETY CHECK ---
    expected_size = n_nodes * n_timesteps * n_features
    actual_size = df[temporal_features].values.size
    
    if expected_size != actual_size:
        raise ValueError(
            f"Dimension Mismatch! Data has {actual_size} points, but you are trying to "
            f"reshape into {n_nodes} nodes * {n_timesteps} steps * {n_features} features "
            f"({expected_size} points). Check for missing rows in your time series!"
        )

    # Reshape features to [Nodes, Time, Features]
    X_temporal = df[temporal_features].values.reshape(n_nodes, n_timesteps, n_features)
    X_tensor = torch.tensor(X_temporal, dtype=torch.float32)

    # Reshape target to [Nodes, Time]
    y_target = df["y"].values.reshape(n_nodes, n_timesteps)
    y_tensor = torch.tensor(y_target, dtype=torch.float32)

    if torch.isnan(X_tensor).any() or torch.isinf(X_tensor).any():
        raise ValueError("CRITICAL: X_tensor contains NaNs or Infs! Check your upstream scaling.")
    
    # Identify Future Covariates
    
    futr_exog = [
        "price", "promo_depth", "is_weekend", "month_name",
        "snap_CA", "snap_TX", "snap_WI", "is_event", "days_until_next_event"
    ]
    # Find the exact integer indices of these features inside your temporal_features list
    futr_indices = [temporal_features.index(f) for f in futr_exog if f in temporal_features]
    
    # 5. Save the PyTorch Dictionary payload
    tensor_path = os.path.join(processed_dir, "stgnn_tensors.pt")
    torch.save({
        "X": X_tensor,
        "y": y_tensor,
        "adj": adj_tensor,
        "n_nodes": n_nodes,
        "n_features": n_features,
        "futr_indices": futr_indices
    }, tensor_path)
    
    print(f"Successfully built Tensors: X {X_tensor.shape}, adj {adj_tensor.shape}")
    print(f"Saved payload to {tensor_path}")


@asset(deps=["prepare_stgnn_tensors"])
def train_hybrid_stgnn() -> None:
    """Trains our custom Spatio-Temporal Graph-MLP model."""
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    from torch.utils.data import DataLoader
    from .hybrid_model import STGNNMixer, LitSTGNNMixer, GraphTimeSeriesDataset

    # 1. Load the payload
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
    X, y, adj = payload["X"], payload["y"], payload["adj"]
    n_nodes, n_features = payload["n_nodes"], payload["n_features"]
    futr_indices = payload["futr_indices"]
    n_futr_features = len(futr_indices)

    # 2. Hyperparameters
    seq_len = 56   # 56 days of history
    pred_len = 28  # 28 days forecast
    hidden_features = 128
    batch_size = 64

    # 3. Temporal Train/Validation Split
    # We hold out the exact final 28 days for validation to match the TSMixer baseline
    val_split_idx = X.shape[1] - pred_len
    
    X_train, y_train = X[:, :val_split_idx, :], y[:, :val_split_idx]
    
    # We only need enough validation data to create a single window evaluating the last 28 days
    X_val = X[:, -seq_len - pred_len:, :]
    y_val = y[:, -seq_len - pred_len:]

    # 4. Initialize DataLoaders
    train_dataset = GraphTimeSeriesDataset(X_train, y_train, seq_len, pred_len, futr_indices)
    val_dataset = GraphTimeSeriesDataset(X_val, y_val, seq_len, pred_len, futr_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 5. Initialize the Custom Hybrid Architecture
    core_model = STGNNMixer(
        seq_len=seq_len, 
        pred_len=pred_len, 
        n_nodes=n_nodes, 
        in_features=n_features, 
        hidden_features=hidden_features,
        n_futr_features=n_futr_features
    )
    
    lit_model = LitSTGNNMixer(model=core_model, adj_matrix=adj, learning_rate=1e-3)

    # 6. Ignite PyTorch Lightning
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=100, 
        callbacks=[early_stop_callback], 
        accelerator="auto" # Will automatically detect and use your RTX GPU!
    )

    print(f"Igniting Custom STGNNMixer on {n_nodes} nodes with multi-relational spatial routing...")
    trainer.fit(lit_model, train_loader, val_loader)
    
    # 7. Save the trained weights
    model_dir = os.path.join(os.getcwd(), "data", "models", "stgnnmixer")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(lit_model.state_dict(), os.path.join(model_dir, "weights.pt"))
    print(f"STGNNMixer explicitly trained and saved to {model_dir}")


@asset(deps=["prepare_ml_data"])
def train_tsmixerx() -> None:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TSMixerx
    
    # 1. Load the ML-ready data
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    
    # 2. Extract configuration limits
    n_series = df["unique_id"].nunique()
    
    # 3. Categorize Exogenous Variables
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    futr_exog = [
        "price", "promo_depth", "is_weekend", "month_name",
        "snap_CA", "snap_TX", "snap_WI", "is_event", "days_until_next_event"
    ]
    
    hist_exog = [col for col in df.columns if 'lag' in col or 'mean' in col or 'std' in col or 'roll_sum' in col]

    # 4. Initialize the Implicit Spatial Model
    model = TSMixerx(
        h=28,                  
        input_size=28 * 2,     
        n_series=n_series,     
        stat_exog_list=stat_exog,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        scaler_type='standard',
        max_steps=100,         # Fast execution for your local benchmark
        early_stop_patience_steps=3,
    )
    
    # 5. Build and train the pipeline
    nf = NeuralForecast(models=[model], freq='D')

    # Grab the unique_id and static columns, and drop all the duplicate daily rows
    static_cols = ["unique_id"] + stat_exog
    static_df = df[static_cols].drop_duplicates()
    
    print(f"Igniting TSMixerx on {n_series} interconnected time series...")
    nf.fit(df=df, static_df=static_df, val_size=28)
    
    # 6. Save the trained artifact
    model_dir = os.path.join(os.getcwd(), "data", "models", "tsmixerx")
    os.makedirs(model_dir, exist_ok=True)
    nf.save(path=model_dir, model_index=None, overwrite=True, save_dataset=False)
    print(f"TSMixerx baseline successfully saved to {model_dir}")


