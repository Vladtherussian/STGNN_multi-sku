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
from window_ops.rolling import rolling_mean, rolling_std


from .hybrid_model import VanillaSTGNN, VanillaSTGNNBlock
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
    sales_train_path = os.path.join(download_m5_data, "sales_train_validation.csv")
    # sales_train_eval_path = os.path.join(download_m5_data, "sales_train_evaluation.csv")

    prices_df= pd.read_csv(prices_path)
    calendar_df= pd.read_csv(calendar_path)
    sales_train_df= pd.read_csv(sales_train_path)
    # sales_train_eval_df= pd.read_csv(sales_train_eval_path)

    if config.downsample_dataset:
        # reduce to only Foods and TX of first 100 per dept, per store
        # Gives exactly 1,437 nodes. Fits perfectly in 10GB VRAM.
        # Cross-Departmental
        sales_train_df = sales_train_df.query("store_id == 'TX_1' & cat_id == 'FOODS'")

        # Gives exactly 1,695 nodes. (565 Hobbies items x 3 Texas Stores)
        # Geographical Supply
        # sales_train_df = sales_train_df.query("cat_id == 'HOBBIES' & state_id == 'TX'")

        # Gives exactly 3,049 nodes. (rtx4090)
        # sales_train_df = sales_train_df.query("store_id in ['TX_1']")

        print("Number of selected items in evaluation set:", len(sales_train_df))

    return prices_df, calendar_df, sales_train_df

@asset(deps=["load_sales_data"])
def process_raw_data(load_sales_data: tuple) -> None:
    """Processes the raw data into a single dataframe."""
    prices_df, calendar_df, sales_train_df = load_sales_data

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
        sales_train_df,
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
    # 1. Base Lags
    #----------------------------------
    # Create Lag 1 through 14 for the Deep Learning models
    for lag_horizon in [1, 2, 3, 7, 14]:
        feature_name = f"sales_lag_{lag_horizon}"
        raw_feature_sales[feature_name] = raw_feature_sales.groupby(["id", "item_id", "store_id"])["sales"].shift(lag_horizon)
        feature_list.append(feature_name)
        
    # Create Safe Lags for LightGBM
    for lag_horizon in [28, 35, 42, 56]: 
        feature_name = f"sales_lag_{lag_horizon}"
        raw_feature_sales[feature_name] = raw_feature_sales.groupby(["id", "item_id", "store_id"])["sales"].shift(lag_horizon)
        feature_list.append(feature_name)

    #----------------------------------
    # 2. Rolling Windows
    #----------------------------------
    for win_day in [7, 14, 28]:
        # Track A: Deep Learning Rolling Stats (Base = Lag 1)
        dl_rolling = raw_feature_sales.groupby(["id", "item_id", "store_id"])["sales_lag_1"].rolling(window=win_day)
        raw_feature_sales[f"dl_sales_mean_{win_day}"] = dl_rolling.mean().reset_index(level=[0, 1, 2], drop=True)
        raw_feature_sales[f"dl_sales_std_{win_day}"] = dl_rolling.std().reset_index(level=[0, 1, 2], drop=True)
        feature_list.extend([f"dl_sales_mean_{win_day}", f"dl_sales_std_{win_day}"])

        # Track B: LightGBM Rolling Stats (Base = Lag 28)
        lgb_rolling = raw_feature_sales.groupby(["id", "item_id", "store_id"])["sales_lag_28"].rolling(window=win_day)
        raw_feature_sales[f"lgb_sales_mean_{win_day}"] = lgb_rolling.mean().reset_index(level=[0, 1, 2], drop=True)
        raw_feature_sales[f"lgb_sales_std_{win_day}"] = lgb_rolling.std().reset_index(level=[0, 1, 2], drop=True)
        feature_list.extend([f"lgb_sales_mean_{win_day}", f"lgb_sales_std_{win_day}"])

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
    categorical_cols = static_metadata
    le = LabelEncoder()
    for col in categorical_cols:
        # Fill NaN categories before encoding
        df[col] = df[col].astype(object).fillna("unknown").astype(str)
        df[col] = le.fit_transform(df[col])

    # One-Hot Encode the temporal categories so the math stays linear
    df = pd.get_dummies(df, columns=["month_name", "weekday"], drop_first=True, dtype=int) 

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

    # Find the cutoff date so we don't leak test data stats into the scaler
    pred_len = 28
    all_dates = sorted(df["ds"].unique())
    val_start_date = all_dates[-(pred_len * 2)]
    
    # Create a mask for the training data
    train_mask = df["ds"] < val_start_date

    scaler = StandardScaler()
    
    # FIT the scaler ONLY on the training data!
    scaler.fit(df.loc[train_mask, continuous_features])
    
    # TRANSFORM the entire dataframe using the training stats
    df[continuous_features] = scaler.transform(df[continuous_features])

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

    # --- NEW: Calculate Historical Volume ---
    # Find the cutoff date so we only calculate volume from the training set
    pred_len = 28
    all_dates = sorted(df["ds"].unique())
    val_start_date = all_dates[-(pred_len * 2)]
    
    # Get the mean sales per item from the training period
    train_df = df[df["ds"] < val_start_date]
    volumes = train_df.groupby("unique_id")["y"].mean().reset_index(name="hist_volume")
    
    # 2. Extract the Static DataFrame for the Adjacency Matrix
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"] 
    static_df = df[["unique_id"] + stat_exog].drop_duplicates().sort_values("unique_id")
    
    # Merge the volume data into the static_df
    static_df = static_df.merge(volumes, on="unique_id", how="left").fillna(0)
    
    # 3. Call your custom function to build the volume-weighted graph!
    # from .utils import build_weighted_adjacency
    adj_tensor = build_weighted_adjacency(static_df=static_df, stat_exog_cols=stat_exog, volume_col="hist_volume")

    # ---------------------------------------------------------
    # NEW: Leak-Proof Department Aggregate Feature
    # ---------------------------------------------------------
    # 1. Calculate the raw average daily sales for each department
    dept_daily = df.groupby(["dept_id", "ds"])["y"].mean().reset_index(name="dept_daily_mean")
    
    # 2. Sort perfectly by time to ensure rolling works correctly
    dept_daily = dept_daily.sort_values(by=["dept_id", "ds"])
    
    # 3. Apply the rolling mean, shifting by 1 to prevent data leakage
    dept_daily["dept_sales_mean_28"] = dept_daily.groupby("dept_id")["dept_daily_mean"].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=1).mean()
    )
    
    # 4. Fill the NaN created by the shift(1) with 0
    dept_daily["dept_sales_mean_28"] = dept_daily["dept_sales_mean_28"].fillna(0)
    
    # 5. Merge this safe, shifted feature back into your main flat dataframe
    df = df.merge(dept_daily[["dept_id", "ds", "dept_sales_mean_28"]], on=["dept_id", "ds"], how="left")
    # ---------------------------------------------------------

    
    # 4. Reshape Temporal Features into 3D Tensors
    n_nodes = df["unique_id"].nunique()
    n_timesteps = df["ds"].nunique()
    
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Drop the LightGBM specific features so they don't enter the graph
    lgb_cols = [col for col in df_numeric.columns if col.startswith("lgb_") or col in ["sales_lag_28", "sales_lag_35", "sales_lag_42", "sales_lag_56"]]
    df_numeric = df_numeric.drop(columns=lgb_cols)
    
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
    base_futr_exog = [
        "price", "promo_depth", "is_weekend", 
        "snap_CA", "snap_TX", "snap_WI", "is_event", "days_until_next_event"
    ]
    
    # Dynamically grab all the one-hot encoded calendar columns
    dummy_calendar_cols = [
        col for col in temporal_features 
        if col.startswith("month_name_") or col.startswith("weekday_")
    ]
    
    # Combine the lists
    futr_exog = base_futr_exog + dummy_calendar_cols
    
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

@asset(deps=["prepare_ml_data"])
def train_lightgbm_baseline() -> None:
    """Trains a Recursive LightGBM using MLForecast to fix the 28-day blindspot."""
    import pandas as pd
    import os
    import lightgbm as lgb
    from mlforecast import MLForecast
    from window_ops.rolling import rolling_mean, rolling_std
    
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    
    pred_len = 28
    all_dates = sorted(df["ds"].unique())
    # NEW CORRECT WAY: LightGBM stops training 56 days before the end, exactly like the STGNN
    # val_start_date = all_dates[-(pred_len * 2)]
    test_start_date = all_dates[-pred_len]
    
    df_train = df[df["ds"] < test_start_date].copy()
    # df_test = df[df["ds"] >= test_start_date].copy()
    
    # 1. Define Exogenous Features
    futr_exog = ["price", "promo_depth", "is_weekend", "snap_CA", "snap_TX", "snap_WI", "is_event", "days_until_next_event"]
    futr_exog = futr_exog + [col for col in df.columns if col.startswith("month_name_") or col.startswith("weekday_")]
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    # We ONLY keep the raw target, dates, and exogenous features. 
    # We let MLForecast build the lags recursively so there are zero data leaks!
    keep_cols = ['unique_id', 'ds', 'y'] + futr_exog + stat_exog
    df_train = df_train[keep_cols].copy()

    
    # 2. Initialize the raw LightGBM Model
    model = lgb.LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=1.5,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    # 3. Wrap it in MLForecast 
    # This automatically recreates your feature_engineering asset, but does it dynamically during inference!
    fcst = MLForecast(
        models={'LightGBM': model},
        freq='D',
        lags=[1, 2, 3, 7, 14], # Re-adds the recent lags you had to drop previously!
        lag_transforms={
            1: [(rolling_mean, 7), (rolling_std, 7), 
                (rolling_mean, 14), (rolling_std, 14),
                (rolling_mean, 28), (rolling_std, 28)]
        }
    )
    
    print("Igniting Recursive LightGBM Baseline...")
    fcst.fit(df_train, id_col='unique_id', time_col='ds', target_col='y', static_features=stat_exog)
    
    # THE FIX: We only need to predict the final 28 days (h=28)
    df_future = df[df["ds"] >= test_start_date][['unique_id', 'ds'] + futr_exog].copy()
    
    predictions = fcst.predict(h=pred_len, X_df=df_future)

    out_dir = os.path.join(os.getcwd(), "data", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    predictions.to_parquet(os.path.join(out_dir, "lgb_predictions.parquet"), index=False)
    print("Recursive LightGBM baseline predictions saved.")


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
    hidden_features = 64
    batch_size = 8

    # 3. Temporal Train/Validation/Test Split
    test_start_idx = X.shape[1] - pred_len
    val_start_idx = test_start_idx - pred_len
    
    # Train data: Everything up to the validation window
    X_train, y_train = X[:, :val_start_idx, :], y[:, :val_start_idx]
    
    # Validation data: Needs `seq_len` of history prior to `val_start_idx`
    X_val = X[:, val_start_idx - seq_len : test_start_idx, :]
    y_val = y[:, val_start_idx - seq_len : test_start_idx]

    # Test data: Needs `seq_len` of history prior to `test_start_idx`
    X_test = X[:, test_start_idx - seq_len :, :]
    y_test = y[:, test_start_idx - seq_len :]

    # 4. Initialize DataLoaders
    train_dataset = GraphTimeSeriesDataset(X_train, y_train, seq_len, pred_len, futr_indices)
    val_dataset = GraphTimeSeriesDataset(X_val, y_val, seq_len, pred_len, futr_indices)
    test_dataset = GraphTimeSeriesDataset(X_test, y_test, seq_len, pred_len, futr_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

    # 6. Ignite PyTorch Lightning & Evaluate
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=100, 
        callbacks=[early_stop_callback], 
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        # accumulate_grad_batches=2 # <--- Tells the optimizer to wait for 2 batches (4+4=8) before updating weights
    )

    print(f"Igniting Custom STGNNMixer on {n_nodes} nodes...")
    trainer.fit(lit_model, train_loader, val_loader)
    
    # NEW: Run the final blind benchmark!
    print("\n" + "="*50)
    print("🚀 RUNNING FINAL BENCHMARK ON UNSEEN TEST SET 🚀")
    print("="*50)
    trainer.test(lit_model, test_loader)
    
    # 7. Save the trained weights
    model_dir = os.path.join(os.getcwd(), "data", "models", "stgnnmixer")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(lit_model.state_dict(), os.path.join(model_dir, "weights.pt"))
    print(f"STGNNMixer explicitly trained and saved to {model_dir}")


@asset(deps=["prepare_ml_data", "prepare_stgnn_tensors"])
def train_residual_stgnn() -> None:
    """Trains the STGNN to specifically predict LightGBM's cross-validation residuals."""
    import os
    import torch
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    from torch.utils.data import DataLoader
    from mlforecast import MLForecast
    
    # Import our new Residual Lightning Module
    from .hybrid_model import STGNNMixer, LitResidualSTGNN, GraphTimeSeriesDataset
    
    # 1. Load Data
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
    
    X, adj = payload["X"], payload["adj"]
    n_nodes, n_features = payload["n_nodes"], payload["n_features"]
    futr_indices = payload["futr_indices"]
    
    # --- THE PERFORMANCE HACKS ---
    seq_len = 14         # Slashed from 56 (Graph only needs recent error shocks)
    pred_len = 28
    hidden_features = 32 # Slashed from 64 (Residuals require less capacity)
    batch_size = 32      # Increased from 8 (Maxes out the RTX 3080)
    
    all_dates = sorted(df["ds"].unique())
    test_start_date = all_dates[-pred_len]
    df_train = df[df["ds"] < test_start_date].copy()

    # ---------------------------------------------------------
    # THE FIX: Enforce exact feature parity with the baseline!
    # ---------------------------------------------------------
    futr_exog = ["price", "promo_depth", "is_weekend", "snap_CA", "snap_TX", "snap_WI", "is_event", "days_until_next_event"]
    futr_exog = futr_exog + [col for col in df.columns if col.startswith("month_name_") or col.startswith("weekday_")]
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    keep_cols = ['unique_id', 'ds', 'y'] + futr_exog + stat_exog
    df_train = df_train[keep_cols].copy()
    
    # 2. Run MLForecast Cross-Validation
    model = lgb.LGBMRegressor(
        objective='tweedie', 
        tweedie_variance_power=1.5, 
        n_estimators=500, 
        learning_rate=0.05, 
        num_leaves=63, 
        n_jobs=-1, 
        random_state=42, 
        verbose=-1
    )
   
    fcst = MLForecast(
            models={'LightGBM': model},
            freq='D',
            lags=[1, 2, 3, 7, 14], # Re-adds the recent lags you had to drop previously!
            lag_transforms={
                1: [(rolling_mean, 7), (rolling_std, 7), 
                    (rolling_mean, 14), (rolling_std, 14),
                    (rolling_mean, 28), (rolling_std, 28)]
        }
    )
    
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    print("Generating Out-Of-Fold Residuals via LightGBM CV...")
    
    # Using your exact snippet to get 140 days (5 windows * 28 days) of clean residuals
    cv_results = fcst.cross_validation(
        df=df_train,
        h=pred_len,
        n_windows=5, 
        step_size=pred_len,
        id_col='unique_id', time_col='ds', target_col='y', static_features=stat_exog
    )
    
    cv_results['residual'] = cv_results['y'] - cv_results['LightGBM']
    
    # 3. Pivot Residuals to match Tensor Shape [Nodes, Time]
    cv_residuals = cv_results[['unique_id', 'ds', 'residual']]
    residual_pivot = cv_residuals.pivot(index='unique_id', columns='ds', values='residual')
    
    # 4. TENSOR ALIGNMENT MAGIC
    # Find the exact 140 days in the main dataset, and slice the X tensor to match!
    cv_dates = sorted(cv_residuals['ds'].unique())
    all_dates_list = list(all_dates)
    start_idx = all_dates_list.index(cv_dates[0])
    end_idx = all_dates_list.index(cv_dates[-1]) + 1
    
    X_cv = X[:, start_idx:end_idx, :]
    y_res_cv = torch.tensor(residual_pivot.values, dtype=torch.float32)
    
    print(f"Aligned Tensors -> X: {X_cv.shape}, Residuals: {y_res_cv.shape}")
    
    # 5. Temporal Train/Val Split (on the 140 days of residuals)
    val_split_idx = X_cv.shape[1] - pred_len
    X_train, y_train = X_cv[:, :val_split_idx, :], y_res_cv[:, :val_split_idx]
    X_val, y_val = X_cv[:, -seq_len - pred_len:, :], y_res_cv[:, -seq_len - pred_len:]
    
    train_dataset = GraphTimeSeriesDataset(X_train, y_train, seq_len, pred_len, futr_indices)
    val_dataset = GraphTimeSeriesDataset(X_val, y_val, seq_len, pred_len, futr_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 6. Ignite the Hybrid Model
    core_model = STGNNMixer(
        seq_len=seq_len, 
        pred_len=pred_len, 
        n_nodes=n_nodes, 
        in_features=1,  # <--- THE FIX: The model now expects ONLY 1 historical feature (the residual)!
        hidden_features=hidden_features, 
        n_futr_features=len(futr_indices)
    )
    
    lit_model = LitResidualSTGNN(model=core_model, adj_matrix=adj)
    
    # NOTE: PyTorch 2.0 Compilation (Uncomment if you transition to a Linux environment. It is finicky on Windows!)
    # lit_model = torch.compile(lit_model) 
    
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    trainer = pl.Trainer(max_epochs=50, callbacks=[early_stop_callback], accelerator="auto", precision="16-mixed")
    
    print("Igniting Residual STGNN Engine...")
    trainer.fit(lit_model, train_loader, val_loader)
    
    # Save Weights
    model_dir = os.path.join(os.getcwd(), "data", "models", "residual_stgnn")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(lit_model.state_dict(), os.path.join(model_dir, "weights.pt"))
    print(f"Residual Model saved to {model_dir}")



# @asset(deps=["prepare_stgnn_tensors"])
# def train_vanilla_stgnn() -> None:
#     """Trains the Vanilla STGNN baseline to compare against the Mixer."""
#     import torch
#     import pytorch_lightning as pl
#     from pytorch_lightning.callbacks import EarlyStopping
#     from torch.utils.data import DataLoader
#     import os
    
#     # NEW: Importing the FULL model (VanillaSTGNN) and your Dataset
#     from .hybrid_model import VanillaSTGNN, LitSTGNNMixer, GraphTimeSeriesDataset

#     # 1. Load the payload
#     processed_dir = os.path.join(os.getcwd(), "data", "processed")
#     payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
#     X, y, adj = payload["X"], payload["y"], payload["adj"]
#     n_nodes, n_features = payload["n_nodes"], payload["n_features"]
#     futr_indices = payload["futr_indices"]

#     # 2. Hyperparameters
#     seq_len = 56   
#     pred_len = 28  
#     hidden_features = 64
#     batch_size = 8

#     # 3. Temporal Train/Validation Split
#     test_start_idx = X.shape[1] - pred_len
#     val_start_idx = test_start_idx - pred_len
    
#     X_train, y_train = X[:, :val_start_idx, :], y[:, :val_start_idx]
#     X_val = X[:, val_start_idx - seq_len : test_start_idx, :]
#     y_val = y[:, val_start_idx - seq_len : test_start_idx]

#     # 4. Initialize DataLoaders
#     train_dataset = GraphTimeSeriesDataset(X_train, y_train, seq_len, pred_len, futr_indices)
#     val_dataset = GraphTimeSeriesDataset(X_val, y_val, seq_len, pred_len, futr_indices)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # 5. Instantiate the FULL Vanilla Model (The "House")
#     core_vanilla_model = VanillaSTGNN(
#         seq_len=seq_len,
#         pred_len=pred_len,
#         n_nodes=n_nodes,
#         in_features=n_features,  
#         hidden_features=hidden_features,
#         n_blocks=3,
#         n_futr_features=len(futr_indices)               
#     )

#     # Wrap it in PyTorch Lightning
#     vanilla_model = LitSTGNNMixer(
#         model=core_vanilla_model, 
#         adj_matrix=adj, 
#         learning_rate=1e-3
#     )
    
#     # 6. Train it
#     early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
#     trainer = pl.Trainer(
#         max_epochs=100,
#         callbacks=[early_stop_callback],
#         accelerator="auto",
#         precision="16-mixed",
#         gradient_clip_val=1.0

#     )
    
#     print(f"Igniting Vanilla STGNN Baseline on {n_nodes} nodes...")
#     trainer.fit(vanilla_model, train_loader, val_loader)
    
#     # 7. Save the weights
#     model_dir = os.path.join(os.getcwd(), "data", "models", "vanilla_stgnn")
#     os.makedirs(model_dir, exist_ok=True)
#     torch.save(vanilla_model.state_dict(), os.path.join(model_dir, "weights.pt"))
#     print(f"Vanilla STGNN explicitly trained and saved to {model_dir}")


# @asset(deps=["prepare_stgnn_tensors"])
# def train_ablation_no_graph() -> None:
#     """Ablation 1: STGNN with Spatial Routing completely disabled (Identity Matrix)."""
#     import torch
#     import pytorch_lightning as pl
#     from pytorch_lightning.callbacks import EarlyStopping
#     from torch.utils.data import DataLoader
#     import os
#     from .hybrid_model import STGNNMixer, LitSTGNNMixer, GraphTimeSeriesDataset

#     # 1. Load the payload
#     processed_dir = os.path.join(os.getcwd(), "data", "processed")
#     payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
#     X, y, adj = payload["X"], payload["y"], payload["adj"]
#     n_nodes, n_features = payload["n_nodes"], payload["n_features"]
#     futr_indices = payload["futr_indices"]
#     n_futr_features = len(futr_indices)

#     # 2. Hyperparameters & Splits
#     seq_len = 56   
#     pred_len = 28  
#     hidden_features = 128
#     batch_size = 32

#     val_split_idx = X.shape[1] - pred_len
#     X_train, y_train = X[:, :val_split_idx, :], y[:, :val_split_idx]
#     X_val = X[:, -seq_len - pred_len:, :]
#     y_val = y[:, -seq_len - pred_len:]

#     train_dataset = GraphTimeSeriesDataset(X_train, y_train, seq_len, pred_len, futr_indices)
#     val_dataset = GraphTimeSeriesDataset(X_val, y_val, seq_len, pred_len, futr_indices)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # 3. Initialize Model with "no_graph"
#     core_model = STGNNMixer(
#         seq_len=seq_len, pred_len=pred_len, n_nodes=n_nodes, 
#         in_features=n_features, hidden_features=hidden_features, 
#         n_futr_features=n_futr_features, 
#         ablation_mode="no_graph"
#     )
    
#     lit_model = LitSTGNNMixer(model=core_model, adj_matrix=adj, learning_rate=1e-3)
#     trainer = pl.Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")], accelerator="auto", precision="16-mixed")

#     print(f"Igniting No-Graph Ablation on {n_nodes} nodes...")
#     trainer.fit(lit_model, train_loader, val_loader)
    
#     # 4. Save to a specific folder
#     out_dir = os.path.join(os.getcwd(), "data", "models", "ablation_no_graph")
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save(lit_model.state_dict(), os.path.join(out_dir, "weights.pt"))


# @asset(deps=["prepare_stgnn_tensors"])
# def train_ablation_static_graph() -> None:
#     """Ablation 2: STGNN restricted to the human-engineered Walmart hierarchy."""
#     import torch
#     import pytorch_lightning as pl
#     from pytorch_lightning.callbacks import EarlyStopping
#     from torch.utils.data import DataLoader
#     import os
#     from .hybrid_model import STGNNMixer, LitSTGNNMixer, GraphTimeSeriesDataset

#     processed_dir = os.path.join(os.getcwd(), "data", "processed")
#     payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
#     X, y, adj = payload["X"], payload["y"], payload["adj"]
#     n_nodes, n_features = payload["n_nodes"], payload["n_features"]
#     futr_indices = payload["futr_indices"]
#     n_futr_features = len(futr_indices)

#     seq_len, pred_len, hidden_features, batch_size = 56, 28, 128, 32

#     val_split_idx = X.shape[1] - pred_len
#     X_train, y_train = X[:, :val_split_idx, :], y[:, :val_split_idx]
#     X_val = X[:, -seq_len - pred_len:, :]
#     y_val = y[:, -seq_len - pred_len:]

#     train_dataset = GraphTimeSeriesDataset(X_train, y_train, seq_len, pred_len, futr_indices)
#     val_dataset = GraphTimeSeriesDataset(X_val, y_val, seq_len, pred_len, futr_indices)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # Initialize Model with "static_graph"
#     core_model = STGNNMixer(
#         seq_len=seq_len, pred_len=pred_len, n_nodes=n_nodes, 
#         in_features=n_features, hidden_features=hidden_features, 
#         n_futr_features=n_futr_features, 
#         ablation_mode="static_graph"
#     )
    
#     lit_model = LitSTGNNMixer(model=core_model, adj_matrix=adj, learning_rate=1e-3)
#     trainer = pl.Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")], accelerator="auto", precision="16-mixed")

#     print(f"Igniting Static-Graph Ablation on {n_nodes} nodes...")
#     trainer.fit(lit_model, train_loader, val_loader)
    
#     out_dir = os.path.join(os.getcwd(), "data", "models", "ablation_static_graph")
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save(lit_model.state_dict(), os.path.join(out_dir, "weights.pt"))

@asset(deps=["prepare_ml_data"])
def train_statistical_baselines() -> None:
    """Trains classic statistical models (AutoARIMA, AutoETS) on the CPU."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS
    
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    
    pred_len = 28
    all_dates = sorted(df["ds"].unique())
    test_start_date = all_dates[-pred_len]
    
    # Isolate to strictly univariate data (No Exogenous Variables)
    df_train = df[df["ds"] < test_start_date][['unique_id', 'ds', 'y']].copy()
    
    # Season length 7 captures weekly retail cycles
    models = [AutoARIMA(season_length=7, stepwise=True, approximation=True), AutoETS(season_length=7)]
    
    sf = StatsForecast(models=models, freq='D', n_jobs=-1)
    
    print("Fitting Statistical Baselines (AutoARIMA, AutoETS)...")
    sf.fit(df=df_train)
    predictions = sf.predict(h=pred_len)
    predictions = predictions.reset_index()
    
    out_dir = os.path.join(os.getcwd(), "data", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    predictions.to_parquet(os.path.join(out_dir, "stats_predictions.parquet"), index=False)
    print("Statistical baseline predictions saved.")

@asset(deps=["prepare_ml_data"])
def train_deep_baselines() -> None:
    """Trains deep learning benchmarks (TSMixerx, TFT, NHITS) on the GPU."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TSMixerx, TFT, NHITS
    
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    
    n_series = df["unique_id"].nunique()
    pred_len = 28
    
    all_dates = sorted(df["ds"].unique())
    test_start_date = all_dates[-pred_len]
    df_train_val = df[df["ds"] < test_start_date].copy()
    df_test = df[df["ds"] >= test_start_date].copy()
    
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    futr_exog = ["price", "promo_depth", "is_weekend", "snap_CA", "snap_TX", "snap_WI", "is_event", "days_until_next_event"]
    futr_exog = futr_exog + [col for col in df.columns if col.startswith("month_name_") or col.startswith("weekday_")]

    # Drop the Deep Learning specific recent lags to prevent leakage
    df_numeric = df.select_dtypes(include=[np.number])
    lgb_cols = [col for col in df_numeric.columns if col.startswith("lgb_") or col in ["sales_lag_28", "sales_lag_35", "sales_lag_42", "sales_lag_56"]]
    df_numeric = df_numeric.drop(columns=lgb_cols)

    hist_exog = [col for col in df_numeric.columns if 'lag' in col or 'mean' in col or 'std' in col or 'roll_sum' in col]

    # Initialize all models (keeping max_steps low for quick execution on your local rig)
    models = [
        TSMixerx(h=pred_len, input_size=pred_len*2, n_series=n_series, stat_exog_list=stat_exog, hist_exog_list=hist_exog, futr_exog_list=futr_exog, max_steps=100),
        TFT(h=pred_len, input_size=pred_len*2, stat_exog_list=stat_exog, hist_exog_list=hist_exog, futr_exog_list=futr_exog, max_steps=100),
        NHITS(h=pred_len, input_size=pred_len*2, stat_exog_list=stat_exog, hist_exog_list=hist_exog, futr_exog_list=futr_exog, max_steps=100)
    ]
    
    nf = NeuralForecast(models=models, freq='D')
    static_df = df_train_val[["unique_id"] + stat_exog].drop_duplicates()
    
    print("Igniting Deep Learning Baselines (TSMixerx, TFT, NHITS)...")
    nf.fit(df=df_train_val, static_df=static_df, val_size=pred_len)
    
    # Generate test predictions
    futr_df = df_test[['unique_id', 'ds'] + futr_exog]
    predictions = nf.predict(df=df_train_val, static_df=static_df, futr_df=futr_df)
    predictions = predictions.reset_index()
    
    out_dir = os.path.join(os.getcwd(), "data", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    predictions.to_parquet(os.path.join(out_dir, "deep_predictions.parquet"), index=False)
    print("Deep learning predictions saved.")

@asset(deps=[
    "train_statistical_baselines", 
    "train_deep_baselines", 
    "train_hybrid_stgnn", 
    "train_ablation_no_graph", 
    "train_ablation_static_graph",
    "train_vanilla_stgnn",
    "train_lightgbm_baseline",
    "train_residual_stgnn"
])
def evaluate_benchmark() -> None:
    import pandas as pd
    import torch
    import os
    from .utils import calculate_m5_metrics
    from .hybrid_model import STGNNMixer, LitSTGNNMixer
    import torch.nn.functional as F
    
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    pred_dir = os.path.join(os.getcwd(), "data", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    # 1. Load ML Ready Data to get the ground truth Test Set
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    pred_len = 28
    all_dates = sorted(df["ds"].unique())
    test_start_date = all_dates[-pred_len]
    train_df = df[df["ds"] < test_start_date].copy()
    test_df = df[df["ds"] >= test_start_date].copy()
    
    # 2. Load the Baseline Predictions
    # stats_preds = pd.read_parquet(os.path.join(pred_dir, "stats_predictions.parquet"))
    # deep_preds = pd.read_parquet(os.path.join(pred_dir, "deep_predictions.parquet"))
    lgb_preds = pd.read_parquet(os.path.join(pred_dir, "lgb_predictions.parquet"))
    all_preds = lgb_preds
    
    # Merge them into a single dataframe
    # all_preds = stats_preds.merge(deep_preds, on=["unique_id", "ds"], how="inner")
    # all_preds = deep_preds.merge(lgb_preds, on=["unique_id", "ds"], how="inner")
    
    # ---------------------------------------------------------
    # 3. Generate STGNN Predictions (Full + Ablations)
    # ---------------------------------------------------------
    payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
    X, y, adj = payload["X"], payload["y"], payload["adj"]
    n_nodes, n_features = payload["n_nodes"], payload["n_features"]
    seq_len = 56
    futr_indices = payload["futr_indices"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_start_idx = X.shape[1] - pred_len
    x_hist = X[:, test_start_idx - seq_len : test_start_idx, :].unsqueeze(0).to(device)
    x_future = X[:, test_start_idx : test_start_idx + pred_len, futr_indices].unsqueeze(0).to(device)
    y_hist = y[:, test_start_idx - seq_len : test_start_idx].unsqueeze(0).to(device)
    
    ordered_unique_ids = sorted(test_df["unique_id"].unique())
    test_dates = sorted(test_df["ds"].unique())

    # Loop through the three variations we trained
    stgnn_models = [
        ("STGNNMixer", "stgnnmixer", "full"),
        # ("STGNN_StaticGraph", "ablation_static_graph", "static_graph"),
        # ("STGNN_NoGraph", "ablation_no_graph", "no_graph")
    ]

    for model_name, folder_name, ablation_mode in stgnn_models:
        weights_path = os.path.join(os.getcwd(), "data", "models", folder_name, "weights.pt")
        
        # If an ablation hasn't been trained yet, skip it gracefully
        if not os.path.exists(weights_path):
            print(f"Skipping {model_name} because weights were not found at {weights_path}.")
            continue
            
        state_dict = torch.load(weights_path)
        hidden_features = state_dict['model.static_node_emb'].shape[1]
        
        core_model = STGNNMixer(
            seq_len=seq_len, pred_len=pred_len, n_nodes=n_nodes, 
            in_features=n_features, hidden_features=hidden_features, 
            n_futr_features=len(futr_indices), ablation_mode=ablation_mode
        )
        
        lit_model = LitSTGNNMixer(model=core_model, adj_matrix=adj.to(device))
        lit_model.load_state_dict(state_dict)
        lit_model.to(device)
        lit_model.eval()
        
        with torch.no_grad():
            # FP32 Armor for Evaluation
            y_hist_fp32 = y_hist.float()
            mean = y_hist_fp32.mean(dim=-1, keepdim=True)
            var = y_hist_fp32.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-5)
            
            # Unpack the Tuple
            y_hat_norm = lit_model(x_hist, x_future)
            
            # Upcast and denormalize
            y_hat_raw = (y_hat_norm.float() * std) + mean
            y_hat = F.softplus(y_hat_raw)
        
        y_hat_cpu = y_hat.squeeze(0).cpu().numpy()  # Shape: [1000, 28]
        
        stgnn_records = []
        for i, uid in enumerate(ordered_unique_ids):
            for j, date in enumerate(test_dates):
                stgnn_records.append({"unique_id": uid, "ds": date, model_name: y_hat_cpu[i, j]})
                
        stgnn_preds = pd.DataFrame(stgnn_records)
        all_preds = all_preds.merge(stgnn_preds, on=["unique_id", "ds"], how="inner")

    # =========================================================
    # Evaluate Vanilla STGNN Baseline
    # =========================================================
    # vanilla_weights_path = os.path.join(os.getcwd(), "data", "models", "vanilla_stgnn", "weights.pt")
    
    # if os.path.exists(vanilla_weights_path):
    #     print("Evaluating Vanilla STGNN Baseline...")
    #     vanilla_state_dict = torch.load(vanilla_weights_path)
        
    #     # Instantiate the "House"
    #     core_vanilla = VanillaSTGNN(
    #         seq_len=seq_len, 
    #         pred_len=pred_len, 
    #         n_nodes=n_nodes, 
    #         in_features=n_features, 
    #         hidden_features=64,  # Same as we trained it with
    #         n_blocks=2,
    #         n_futr_features=len(futr_indices)
    #     )
        
    #     # Wrap it in Lightning and load the weights
    #     lit_vanilla = LitSTGNNMixer(model=core_vanilla, adj_matrix=adj.to(device))
    #     lit_vanilla.load_state_dict(vanilla_state_dict)
    #     lit_vanilla.to(device)
    #     lit_vanilla.eval()
        
    #     with torch.no_grad():
    #         mean = y_hist.mean(dim=-1, keepdim=True)
    #         std = y_hist.std(dim=-1, keepdim=True) + 1e-5
    #         y_hat_norm = lit_vanilla(x_hist, x_future)
    #         # Denormalize
    #         y_hat = (y_hat_norm * std) + mean
            
    #     y_hat_cpu = y_hat.squeeze(0).cpu().numpy()
        
    #     # Format the predictions into a DataFrame
    #     vanilla_records = []
    #     for i, uid in enumerate(ordered_unique_ids):
    #         for j, date in enumerate(test_dates):
    #             vanilla_records.append({"unique_id": uid, "ds": date, "VanillaSTGNN": y_hat_cpu[i, j]})
                
    #     vanilla_preds = pd.DataFrame(vanilla_records)
        
    #     # Merge it into the master predictions dataframe
    #     all_preds = all_preds.merge(vanilla_preds, on=["unique_id", "ds"], how="inner")
    # else:
    #     print("Skipping VanillaSTGNN because weights were not found.")

    # ---------------------------------------------------------
    # 🔥 THE NAIVE ENSEMBLE (50/50 Average) 🔥
    # ---------------------------------------------------------
    if "LightGBM" in all_preds.columns and "STGNNMixer" in all_preds.columns:
        all_preds["Hybrid_Ensemble"] = (all_preds["LightGBM"] * 0.5) + (all_preds["STGNNMixer"] * 0.5)

    # ---------------------------------------------------------
    # 🚀 THE TRUE TWO-STAGE RESIDUAL HYBRID 🚀
    # ---------------------------------------------------------
    residual_weights_path = os.path.join(os.getcwd(), "data", "models", "residual_stgnn", "weights.pt")
    
    if os.path.exists(residual_weights_path) and "LightGBM" in all_preds.columns:
        print("\nEvaluating True Two-Stage Hybrid (LightGBM + Residual STGNN)...")
        from .hybrid_model import LitResidualSTGNN
        
        # Apply the exact performance hacks used during training
        res_seq_len = 14
        
        state_dict = torch.load(residual_weights_path)
        res_hidden_features = state_dict['model.static_node_emb'].shape[1]
        
        core_res_model = STGNNMixer(
            seq_len=res_seq_len, 
            pred_len=pred_len, 
            n_nodes=n_nodes, 
            in_features=1, # The Pure Approach: Only 1 historical feature!
            hidden_features=res_hidden_features, 
            n_futr_features=len(futr_indices)
        )
        
        lit_res_model = LitResidualSTGNN(model=core_res_model, adj_matrix=adj.to(device))
        lit_res_model.load_state_dict(state_dict)
        lit_res_model.to(device)
        lit_res_model.eval()
        
        with torch.no_grad():
            # 1. Slice the last 14 days of raw sales
            y_hist_14 = y[:, test_start_idx - res_seq_len : test_start_idx].unsqueeze(0).to(device)
            
            # 2. INFERENCE HACK: Approximate LightGBM's historical baseline by mean-centering.
            # This turns raw sales into "recent shock residuals" for the graph to route.
            y_hist_14_mean = y_hist_14.mean(dim=-1, keepdim=True)
            pure_x_inference = (y_hist_14 - y_hist_14_mean).unsqueeze(-1).float() # Shape: [1, N, 14, 1]
            
            # 3. Predict the future spatial errors
            # (No FP32 armor, no denormalization, no Softplus! It outputs raw +/- residuals)
            y_hat_residual = lit_res_model(pure_x_inference, x_future)
        
        y_hat_res_cpu = y_hat_residual.squeeze(0).cpu().numpy()
        
        # 4. Format into a dataframe
        res_records = []
        for i, uid in enumerate(ordered_unique_ids):
            for j, date in enumerate(test_dates):
                res_records.append({"unique_id": uid, "ds": date, "STGNN_Residual": y_hat_res_cpu[i, j]})
                
        res_preds = pd.DataFrame(res_records)
        all_preds = all_preds.merge(res_preds, on=["unique_id", "ds"], how="inner")
        
        # 5. THE MAGIC: Add the STGNN spatial error back to the LightGBM temporal baseline!
        all_preds["TwoStage_Hybrid"] = all_preds["LightGBM"] + all_preds["STGNN_Residual"]

    # Force all prediction columns to be 0 or higher
    model_cols = [col for col in all_preds.columns if col not in ['unique_id', 'ds', 'y']]
    all_preds[model_cols] = all_preds[model_cols].clip(lower=0.0)

    # ---------------------------------------------------------
    # 🔥 THE ENSEMBLE HYBRID 🔥
    # ---------------------------------------------------------
    # We take a simple 50/50 average of the two most powerful models.
    # Because they make uncorrelated errors, the average will likely beat both of them natively.
    if "LightGBM" in all_preds.columns and "STGNNMixer" in all_preds.columns:
        all_preds["Hybrid_Ensemble"] = (all_preds["LightGBM"] * 0.5) + (all_preds["STGNNMixer"] * 0.5)
        
        

    # ---------------------------------------------------------
    # 4. Calculate Final Metrics
    # ---------------------------------------------------------
    raw_feature_sales = pd.read_parquet(os.path.join(DATA_DIR, "processed", "raw_feature_sales.parquet"))
    # Sort by timestamp to ensure correct order for lag features
    raw_feature_sales = raw_feature_sales.sort_values(by=["id", "item_id", "store_id", "timestamp"])

    print("\nCalculating rigorous M5 validation metrics...")
    
    # Update the names list to include your ablations
    model_names = ["AutoARIMA", "AutoETS", "TSMixerx", "TFT", "NHITS", "STGNNMixer", 
                   "STGNN_StaticGraph", "STGNN_NoGraph", "VanillaSTGNN", "LightGBM", "Hybrid_Ensemble", "TwoStage_Hybrid" ]
    
    
    
    # Filter the list down to models that actually exist in the dataframe
    model_names = [m for m in model_names if m in all_preds.columns]
    
    results_df, grouped_metrics = calculate_m5_metrics(train_df, test_df, all_preds, model_names, raw_feature_sales, pred_len)
    
    grouped_metrics_pivoted = pd.pivot_table(grouped_metrics, values = "value", index=["dept_name", "Model"], columns="metric", aggfunc="mean").sort_values(by="weighted_rmsse")
    
    print("\n" + "="*80)
    print("🏆 FINAL FORECASTING TOURNAMENT LEADERBOARD 🏆")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")

    print("\n" + "="*80)
    print("📊 CATEGORY-LEVEL METRICS Global Error Attribution (WRMSSE)📊")
    print("="*80)
    print(grouped_metrics_pivoted.to_string())
    print("="*80 + "\n")

    # ---------------------------------------------------------
    # 5. Merge the actual ground truth and save the master file
    # ---------------------------------------------------------
    # Add the actual sales ('y') to our predictions dataframe
    all_preds = all_preds.merge(test_df[['unique_id', 'ds', 'y']], on=["unique_id", "ds"], how="inner")
    
    # Save the master evaluation file for Plotly Visualization
    final_pred_path = os.path.join(pred_dir, "final_eval_predictions.parquet")
    all_preds.to_parquet(final_pred_path, index=False)
    print(f"Master predictions file saved for visualization: {final_pred_path}")

# @asset(deps=["prepare_ml_data"])
# def train_tsmixerx() -> None:
#     from neuralforecast import NeuralForecast
#     from neuralforecast.models import TSMixerx
#     import pandas as pd
#     import os
    
#     processed_dir = os.path.join(os.getcwd(), "data", "processed")
#     df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    
#     n_series = df["unique_id"].nunique()
#     pred_len = 28
    
#     # NEW: Split the dataframe by Date to match the STGNN tensors
#     all_dates = sorted(df["ds"].unique())
#     test_start_date = all_dates[-pred_len]
    
#     # Training + Validation DataFrame (TSMixerx will split this internally using val_size=28)
#     df_train_val = df[df["ds"] < test_start_date].copy()
    
#     # Test DataFrame (Strictly held out)
#     df_test = df[df["ds"] >= test_start_date].copy()
    
#     stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
#     futr_exog = ["price", "promo_depth", "is_weekend", "snap_CA", "snap_TX", "snap_WI", "is_event", "days_until_next_event"]
#     futr_exog = futr_exog + [col for col in df.columns if col.startswith("month_name_") or col.startswith("weekday_")]
    
#     #  Drop the Deep Learning specific recent lags to prevent leakage
#     df_numeric = df.select_dtypes(include=[np.number])
#     lgb_cols = [col for col in df_numeric.columns if col.startswith("lgb_") or col in ["sales_lag_28", "sales_lag_35", "sales_lag_42", "sales_lag_56"]]
#     df_numeric = df_numeric.drop(columns=lgb_cols)

#     hist_exog = [col for col in df_numeric.columns if 'lag' in col or 'mean' in col or 'std' in col or 'roll_sum' in col]

#     model = TSMixerx(
#         h=pred_len,                  
#         input_size=pred_len * 2,     
#         n_series=n_series,     
#         stat_exog_list=stat_exog,
#         hist_exog_list=hist_exog,
#         futr_exog_list=futr_exog,
#         scaler_type='standard',
#         max_steps=100,         
#         early_stop_patience_steps=3,
#     )
    
#     nf = NeuralForecast(models=[model], freq='D')
#     static_df = df_train_val[["unique_id"] + stat_exog].drop_duplicates()
    
#     print(f"Igniting TSMixerx baseline...")
#     # Train only on the pre-test data
#     nf.fit(df=df_train_val, static_df=static_df, val_size=pred_len)
    
#     print("\n" + "="*50)
#     print("🚀 RUNNING BASELINE BENCHMARK ON UNSEEN TEST SET 🚀")
#     print("="*50)
    
#     # Provide the future exogenous covariates so it can forecast the test period
#     futr_df = df_test[['unique_id', 'ds'] + futr_exog]
#     predictions = nf.predict(df=df_train_val, static_df=static_df, futr_df=futr_df)
    
#     # Merge and calculate actual MAE
#     results = predictions.merge(df_test[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
#     mae = (results['TSMixerx'] - results['y']).abs().mean()
#     print(f"TSMixerx Test MAE: {mae:.4f}")
#     print("="*50 + "\n")
    
#     model_dir = os.path.join(os.getcwd(), "data", "models", "tsmixerx")
#     os.makedirs(model_dir, exist_ok=True)
#     nf.save(path=model_dir, model_index=None, overwrite=True, save_dataset=False)


@asset(deps=["train_hybrid_stgnn"])
def explain_forecast_captum() -> None:
    import torch
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from captum.attr import IntegratedGradients
    from .hybrid_model import STGNNMixer, LitSTGNNMixer

    # 1. Setup paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    processed_dir = os.path.join(DATA_DIR, "processed")
    model_dir = os.path.join(DATA_DIR, "models", "stgnnmixer")

    # 2A. Extract Exact Feature Names & Product Metadata
    df = pd.read_parquet(os.path.join(processed_dir, "ml_ready_data.parquet"))
    stat_exog = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ignore_cols = ["unique_id", "ds", "y"] + stat_exog
    df_numeric = df.select_dtypes(include=[np.number])
    feature_names = [col for col in df_numeric.columns if col not in ignore_cols]

    # Create a lookup table for the nodes
    # Load the un-encoded data to get the real string names
    raw_df = pd.read_parquet(os.path.join(processed_dir, "model_input.parquet"))
    
    # Filter to only the items that made it into our final dataset, 
    # and sort them by unique_id so they perfectly match the PyTorch tensor indices.
    valid_ids = df["unique_id"].unique()
    static_df = raw_df[raw_df["unique_id"].isin(valid_ids)][
        ["unique_id", "item_id", "dept_id", "store_id"]
    ].drop_duplicates().sort_values("unique_id").reset_index(drop=True)

    # 2B. Load Tensors & Hyperparameters
    payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
    X, y, adj = payload["X"], payload["y"], payload["adj"]
    n_nodes, n_features = payload["n_nodes"], payload["n_features"]
    
    seq_len = 56
    pred_len = 28
    futr_indices = payload["futr_indices"]
    n_futr_features = len(futr_indices)
    
    weights_path = os.path.join(model_dir, "weights.pt")
    state_dict = torch.load(weights_path)
    hidden_features = state_dict['model.static_node_emb'].shape[1]

    # 3. GPU Acceleration Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_hist = X[:, -seq_len-pred_len : -pred_len, :].unsqueeze(0).to(device)
    x_future = X[:, -pred_len:, futr_indices].unsqueeze(0).to(device)

    core_model = STGNNMixer(seq_len, pred_len, n_nodes, n_features, hidden_features, n_futr_features)
    lit_model = LitSTGNNMixer(model=core_model, adj_matrix=adj.to(device))
    lit_model.load_state_dict(state_dict)
    lit_model.to(device)
    lit_model.eval()

    # Define Target Node
    TARGET_NODE_IDX = 0
    target_metadata = static_df.iloc[TARGET_NODE_IDX]

    class NodeForecastWrapper(torch.nn.Module):
        def __init__(self, model, target_node, target_day):
            super().__init__()
            self.model = model
            self.target_node = target_node
            self.target_day = target_day

        def forward(self, hist, futr):
            out = self.model(hist, futr)
            return out[:, self.target_node, self.target_day]

    wrapper_model = NodeForecastWrapper(lit_model, target_node=TARGET_NODE_IDX, target_day=0)

    baseline_hist = torch.zeros_like(x_hist).to(device)
    baseline_futr = torch.zeros_like(x_future).to(device)

    # 4. Ignite Integrated Gradients
    print(f"Calculating Integrated Gradients for {target_metadata['item_id']} at {target_metadata['store_id']}...")
    ig = IntegratedGradients(wrapper_model)
    
    attributions = ig.attribute(
        inputs=(x_hist, x_future),
        baselines=(baseline_hist, baseline_futr),
        n_steps=50, 
        internal_batch_size=1
    )
    
    attr_hist, attr_futr = attributions

    # Move results back to CPU
    attr_hist_cpu = attr_hist.detach().cpu().numpy()
    attr_futr_cpu = attr_futr.detach().cpu().numpy()
    
    # ---------------------------------------------------------
    # NEW: 5. Spatial (Cross-Item) Influence Calculation
    # ---------------------------------------------------------
    # 1. Magnitude: Who has the biggest overall impact? (Absolute sum)
    magnitude_influence = np.abs(attr_hist_cpu[0]).sum(axis=(1, 2))
    magnitude_influence[TARGET_NODE_IDX] = 0.0 
    
    # 2. Direction: Is it a Halo (+) or Cannibalization (-)? (Raw sum)
    directional_influence = attr_hist_cpu[0].sum(axis=(1, 2))
    
    # Get top 3 most influential external nodes by MAGNITUDE
    top_spatial_indices = np.argsort(magnitude_influence)[-3:][::-1]

    # Node 0's own feature importance (Using axis=0 for NumPy!)
    total_hist_importance = attr_hist_cpu[0, TARGET_NODE_IDX].sum(axis=0)
    total_futr_importance = attr_futr_cpu[0, TARGET_NODE_IDX].sum(axis=0)
    
    print("\n" + "━"*70)
    print("🧠 EXPLAINABLE AI: MULTI-RELATIONAL ATTRIBUTION 🧠")
    print("━"*70)
    print(f"TARGET ITEM: {target_metadata['item_id']} | Dept: {target_metadata['dept_id']} | Store: {target_metadata['store_id']}")
    print("━"*70)
    
    print("\n[1] OWN FEATURE IMPORTANCE (What intrinsic data drove this forecast?)")
    print("--- Future Covariates ---")
    for i, futr_idx in enumerate(futr_indices):
        name = feature_names[futr_idx]
        print(f"{name.ljust(25)}: {total_futr_importance[i]:+.4f}")
        
    print("\n--- Historical Features (Top 3) ---")
    top_hist_indices = np.argsort(np.abs(total_hist_importance))[-3:][::-1]
    for idx in top_hist_indices:
        name = feature_names[idx]
        print(f"{name.ljust(25)}: {total_hist_importance[idx]:+.4f}")

    print("\n[2] SPATIAL INFLUENCE (Which OTHER products drove this forecast?)")
    for rank, idx in enumerate(top_spatial_indices):
        inf_node = static_df.iloc[idx]
        mag_score = magnitude_influence[idx]
        dir_score = directional_influence[idx]
        
        # Determine the economic relationship!
        relationship = "HALO EFFECT (+)" if dir_score > 0 else "CANNIBALIZATION (-)"
        
        print(f"#{rank+1} Influence (Magnitude {mag_score:.4f}):")
        print(f"    Item:  {inf_node['item_id']} | Type: {relationship} ({dir_score:+.4f})")
        print(f"    Store: {inf_node['store_id']} | Dept: {inf_node['dept_id']}")
        
        if inf_node['dept_id'] != target_metadata['dept_id']:
            print("    *CROSS-DEPARTMENT DISCOVERY*")
        
    print("━"*70 + "\n")


# @asset(deps=["train_hybrid_stgnn"])
# def simulate_promotion_shock() -> None:
#     import torch
#     import os
#     from .hybrid_model import STGNNMixer, LitSTGNNMixer
#     from pathlib import Path

#     # 1. Setup paths to guarantee we hit the correct root directory
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent
#     DATA_DIR = os.path.join(PROJECT_ROOT, "data")
#     processed_dir = os.path.join(DATA_DIR, "processed")
#     model_dir = os.path.join(DATA_DIR, "models", "stgnnmixer")

#     # 2. Load the exact ML-ready tensors used for training
#     payload = torch.load(os.path.join(processed_dir, "stgnn_tensors.pt"))
#     X, y, adj = payload["X"], payload["y"], payload["adj"]
#     n_nodes, n_features = payload["n_nodes"], payload["n_features"]

#     # 3. Network Hyperparameters 
#     seq_len = 56
#     pred_len = 28
#     hidden_features = 128  # Keeping the memory-optimized size
    
#     # Identify future covariates (price, promo_depth, events, etc.)
#     # In your previous asset, you defined 9 future features
#     futr_indices = payload["futr_indices"]
#     n_futr_features = len(futr_indices)

#     # 4. Extract a single test window (the last 84 days of the dataset)
#     x_hist = X[:, -seq_len-pred_len : -pred_len, :].unsqueeze(0) # [1, Nodes, 56, Feat]
#     y_hist = y[:, -seq_len-pred_len : -pred_len].unsqueeze(0)    # [1, Nodes, 56]
    
#     # Extract the true, unmodified future events
#     x_future_base = X[:, -pred_len:, futr_indices].unsqueeze(0)  # [1, Nodes, 28, Futr_Feat]

#     # 5. Initialize the Core PyTorch Architecture
#     core_model = STGNNMixer(
#         seq_len=seq_len, 
#         pred_len=pred_len, 
#         n_nodes=n_nodes, 
#         in_features=n_features, 
#         hidden_features=hidden_features,
#         n_futr_features=n_futr_features
#     )
    
#     # Wrap in Lightning and load the trained weights from the RTX 3080
#     lit_model = LitSTGNNMixer(model=core_model, adj_matrix=adj)
#     weights_path = os.path.join(model_dir, "weights.pt")
#     lit_model.load_state_dict(torch.load(weights_path))
#     lit_model.eval() # Lock the model into inference mode

#     # 6. Generate Baseline Forecast (Business as Usual)
#     with torch.no_grad():
#         # Apply your RevIN normalization manually for this single inference
#         mean = y_hist.mean(dim=-1, keepdim=True)
#         std = y_hist.std(dim=-1, keepdim=True) + 1e-5
        
#         y_hat_norm_base = lit_model(x_hist, x_future_base)
#         y_hat_base = (y_hat_norm_base * std) + mean

#     # 7. 💥 THE CAUSAL SHOCK 💥
#     # We clone the future data to create an alternate reality
#     x_future_shock = x_future_base.clone()
    
#     # Target Node 0. Index 1 in the future features is 'promo_depth'
#     # We add 3.0 (which represents a massive 3 standard deviation promotional discount)
#     # to all 28 days in the forecast horizon.
#     x_future_shock[0, 0, :, 1] += 3.0 
    
#     # Generate the Shocked Forecast
#     with torch.no_grad():
#         y_hat_norm_shock = lit_model(x_hist, x_future_shock)
#         y_hat_shock = (y_hat_norm_shock * std) + mean

#     # 8. Calculate the Delta
#     # How many extra (or fewer) units were sold over the 28 days due to the shock?
#     node_0_diff = (y_hat_shock[0, 0, :] - y_hat_base[0, 0, :]).sum().item()
#     node_584_diff = (y_hat_shock[0, 584, :] - y_hat_base[0, 584, :]).sum().item()
#     node_646_diff = (y_hat_shock[0, 646, :] - y_hat_base[0, 646, :]).sum().item()

#     print("\n" + "━"*50)
#     print("💥 CAUSAL PROMOTION SHOCK RESULTS 💥")
#     print("━"*50)
#     print("Action: Artificially applied a massive promotion to Node 0.")
#     print(f"-> Node 0 (Targeted Item) 28-Day Impact:     {node_0_diff:+.2f} units")
#     print(f"-> Node 584 (Strongest Link) Sales Impact:   {node_584_diff:+.2f} units")
#     print(f"-> Node 646 (2nd Strongest Link) Impact:     {node_646_diff:+.2f} units")
    
#     if node_584_diff < 0:
#         print("\nInsight: The graph successfully simulated CANNIBALIZATION.")
#     else:
#         print("\nInsight: The graph successfully simulated a HALO EFFECT.")
#     print("━"*50 + "\n")