import os
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List
import pandas as pd
import gdown

from dagster import asset, Config

# 1. Define a Config class for your parameters
class SalesDataConfig(Config):
    downsample_dataset: bool = True

@asset
def download_m5_data() -> str:
    """Downloads and extracts the M5 dataset, returning the folder path."""
    work_directory = os.getcwd() + "\\data"
    os.makedirs(os.getcwd(), exist_ok=True)

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

@asset
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
        # The items with the most sales.
        selected_items_names = ['FOODS_3_090', 'FOODS_3_586', 'FOODS_3_252', 'FOODS_3_555',
                                'FOODS_3_587', 'FOODS_3_714', 'FOODS_3_694', 'FOODS_3_226',
                                #'FOODS_3_202', 'FOODS_3_120', 'FOODS_3_723', 'FOODS_3_635',
                                #'FOODS_3_808', 'FOODS_3_377', 'FOODS_3_541', 'FOODS_3_080',
                                #'FOODS_3_318', 'FOODS_2_360', 'FOODS_3_681', 'FOODS_3_234',
                            ]

        sampled_sales_train_eval_df = sales_train_eval_df[sales_train_eval_df["item_id"].isin(selected_items_names)]
        print("Number of selected items in evaluation set:", len(sampled_sales_train_eval_df))
        sales_train_eval_df = sampled_sales_train_eval_df

    return prices_df, calendar_df, sales_train_eval_df

@asset
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
    wm_yr_wk_map = cleaned_calendar[["weekday", "timestamp", "wm_yr_wk", "event_name_1", 
                                     "event_type_1", "event_name_2", "event_type_2", 
                                     "snap_CA", "snap_TX", "snap_WI"]]

    #----------------------------------
    # Process Sell Price Data
    #----------------------------------
    cleaned_sell_prices = prices_df.copy()
    cleaned_sell_prices = cleaned_sell_prices.merge(wm_yr_wk_map, on="wm_yr_wk", how="left")
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
    cleaned_sales["day"] = cleaned_sales["day"].apply(lambda x: int(x[2:]))
    origin_date = datetime(2011, 1, 29)
    cleaned_sales["timestamp"] = cleaned_sales["day"].apply(
        lambda x: (origin_date + timedelta(days=x - 1))
        )
    # del cleaned_sales["id"]

    raw_feature_sales = cleaned_sales.merge(cleaned_sell_prices, on=["item_id", "timestamp", "store_id"], how="left")

    # Define path to dump processed data and create directory if it doesn't exist
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    return raw_feature_sales.to_parquet(os.path.join(processed_dir, "raw_feature_sales.parquet"), index=False)
    

@asset(deps=["process_raw_data"])
def feature_engineering() -> pd.DataFrame:
    # List to track all the features we create, so we can easily reference them in the next asset when we train our model.
    feature_list = []

    raw_feature_sales = pd.read_parquet(os.path.join(os.getcwd(), "data", "processed", "raw_feature_sales.parquet"))
    # Sort by timestamp to ensure correct order for lag features
    raw_feature_sales = raw_feature_sales.sort_values(by=["item_id", "store_id", "timestamp"])

    #----------------------------------
    # Lag feature for sales
    #----------------------------------
    for lag_horizon in [1, 2, 3, 7, 14]:
        feature_name = f"sales_lag_{lag_horizon}"
        raw_feature_sales[feature_name] = raw_feature_sales.groupby(["item_id", "store_id"])["sales"].shift(lag_horizon)
        feature_list.append(feature_name)

#----------------------------------
    # Rolling window features (Vectorized)
    #----------------------------------
    for win_day in [7, 14, 28, 84]:
        # Create the grouped rolling object ONCE per window size
        rolling_obj = raw_feature_sales.groupby(["item_id", "store_id"])["sales"].rolling(window=win_day)

        # Calculate the native C-optimized metrics. 
        # groupby.rolling() creates a MultiIndex, so we drop the grouping levels to map it cleanly back to the original rows.
        raw_feature_sales[f"sales_mean_{win_day}"] = rolling_obj.mean().reset_index(level=[0, 1], drop=True)
        raw_feature_sales[f"sales_std_{win_day}"] = rolling_obj.std().reset_index(level=[0, 1], drop=True)
        raw_feature_sales[f"sales_roll_sum_{win_day}"] = rolling_obj.sum().reset_index(level=[0, 1], drop=True)

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
    raw_feature_sales["is_event"] = ((raw_feature_sales["event_name_1"] != "no_event") | (raw_feature_sales["event_name_2"] != "no_event")).astype(int)
    feature_list.append("is_event")

    # Days until next event
    #----------------------------------
    # Create a temporary column that ONLY contains the timestamp if an event is happening today
    raw_feature_sales["next_event_date"] = raw_feature_sales["timestamp"].where(raw_feature_sales["is_event"] == 1)

    # Backward fill the empty rows. This takes the next available event date and drags it UP the column to fill the past dates.
    raw_feature_sales["next_event_date"] = raw_feature_sales.groupby(["item_id", "store_id"])["next_event_date"].bfill()

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
    raw_feature_sales['base_price'] = raw_feature_sales.groupby(['item_id', 'store_id'])['sell_price'].transform(
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
    sampling_once_a_day = sales_per_dept.unique_timestamps()

    # Cumulative 28 days sum of sales, per department.
    sum28_sales_per_dept = sales_per_dept["sales"].moving_sum(
            window_length=tp.duration.days(28),
            sampling=sampling_once_a_day
        )

    # Give it a name for book-keeping.
    sum28_sales_per_dept = sum28_sales_per_dept.prefix("f_sum28_per_dep_")

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

    