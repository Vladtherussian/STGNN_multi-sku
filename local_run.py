from dagster import materialize, load_assets_from_modules, RunConfig, AssetSelection
import src.assets as assets
from src.assets import SalesDataConfig

if __name__ == "__main__":
    print("Discovering assets...")
    all_assets = load_assets_from_modules([assets])
    
    print("Executing the pipeline DAG...")
    
    # 1. Start with your base selection (prepare_stgnn_tensors and everything downstream)
    base_selection = AssetSelection.keys("evaluate_benchmark").downstream()
    
    # 2. Subtract the specific asset you want to exclude
    asset_to_exclude = ["train_statistical_baselines"]
    final_selection = base_selection - AssetSelection.keys(asset_to_exclude)
    
    # Pass the final_selection object into the selection parameter
    result = materialize(
        all_assets,
        selection=final_selection,
        # run_config=RunConfig(
        #     ops={
        #         "load_sales_data": SalesDataConfig(downsample_dataset=True)
        #     }
        # )
    )
    
    if result.success:
        print("\nSuccess! Extracting final dataframe tuple:")
        # Extract the tuple returned by your load_sales_data asset
        # (Note: load_sales_data won't execute if it's upstream of your selection)
        dataframes = result.output_for_node("load_sales_data")
        prices_df = dataframes[0] # Unpack the first item
        print(prices_df.head())