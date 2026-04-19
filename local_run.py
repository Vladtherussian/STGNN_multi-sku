from dagster import materialize, load_assets_from_modules, RunConfig
import src.assets as assets
from src.assets import SalesDataConfig # <-- Import your new config class!

if __name__ == "__main__":
    print("Discovering assets...")
    all_assets = load_assets_from_modules([assets])
    
    print("Executing the pipeline DAG...")
    
    # Pass your parameter using the 'ops' keyword and your config object
    result = materialize(
        all_assets,
        selection=["feature_engineering*"],
        # run_config=RunConfig(
        #     ops={
        #         "load_sales_data": SalesDataConfig(downsample_dataset=True)
        #     }
        # )
    )
    
    if result.success:
        print("\nSuccess! Extracting final dataframe tuple:")
        # Extract the tuple returned by your load_sales_data asset
        dataframes = result.output_for_node("load_sales_data")
        prices_df = dataframes[0] # Unpack the first item
        print(prices_df.head())