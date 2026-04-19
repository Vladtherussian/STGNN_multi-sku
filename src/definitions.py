from dagster import Definitions, load_assets_from_modules
import assets

# To invoke the environment, run the following command in your terminal:
# Invoke-Expression (poetry env activate)
# cd src
# dagster dev -f definitions.py

# This automatically discovers every @asset function in your assets.py file
all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
)