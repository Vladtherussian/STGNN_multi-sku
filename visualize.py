import pandas as pd
import plotly.express as px
import os

def plot_multiple_forecasts(num_items=10):
    # 1. Load the master predictions file
    pred_path = os.path.join("data", "predictions", "final_eval_predictions.parquet")
    
    if not os.path.exists(pred_path):
        print(f"Error: Could not find {pred_path}. Did the Dagster tournament finish?")
        return

    df = pd.read_parquet(pred_path)

    # 2. Select the items to visualize
    # Grab the first 10 unique items (you can randomize this or slice it differently)
    unique_items = df['unique_id'].unique()
    target_items = unique_items[1000:1010]

    # Filter down to just our targeted SKUs
    plot_df = df[df['unique_id'].isin(target_items)]

    # 3. Reshape the data for Plotly Express
    # Notice we keep 'unique_id' as an id_var so Plotly knows how to split the facets
    models_to_plot = ['y', 
                    #   'AutoARIMA', 'AutoETS', 
                    #   'TSMixerx', 'TFT', 'NHITS', 
                      'STGNNMixer', 'LightGBM',
                    #   'STGNN_StaticGraph','STGNN_NoGraph'
                      ]
    
    plot_df_melted = plot_df.melt(
        id_vars=['ds', 'unique_id'], 
        value_vars=models_to_plot, 
        var_name='Model', 
        value_name='Sales'
    )

    # 4. Build the Interactive Faceted Plot
    fig = px.line(
        plot_df_melted, 
        x='ds', 
        y='Sales', 
        color='Model',
        facet_col='unique_id',      # Split by item ID
        facet_col_wrap=2,           # 2 columns wide
        title=f'Forecasting Tournament Results: Sample of {num_items} Items',
        labels={'ds': 'Date', 'Sales': 'Units Sold', 'Model': 'Predictor'},
        color_discrete_map={
            'y': 'black',
            'STGNNMixer': 'red',
            # 'AutoARIMA': 'lightblue',
            # 'AutoETS': 'lightgreen',
            'STGNN_NoGraph': 'green',
            'STGNN_StaticGraph': 'cyan',
            'LightGBM': 'blue',
            'TSMixerx': 'orange',
            'TFT': 'purple',
            'NHITS': 'pink'
        },
        height=1500  # Make the canvas very tall to fit all the charts comfortably
    )

    # Allow each subplot to have its own dynamic Y-axis scale
    fig.update_yaxes(matches=None)
    # Give the subplot titles a cleaner look (removes the "unique_id=" prefix)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # 5. Format Line Thickness
    # In faceted plots, we update traces by their specific name selector
    fig.update_traces(patch={"line": {"width": 4}}, selector={"name": "y"})
    fig.update_traces(patch={"line": {"width": 3}}, selector={"name": "STGNNMixer"})
    
    # Keep baselines thin and hidden by default to avoid visual clutter
    baselines = ['AutoARIMA', 'AutoETS', 'TSMixerx', 'TFT', 'NHITS']
    for base in baselines:
        fig.update_traces(patch={"line": {"width": 1.5}, "visible": "legendonly"}, selector={"name": base})

    # 6. Launch in the Browser!
    fig.show()

if __name__ == "__main__":
    plot_multiple_forecasts(num_items=30)