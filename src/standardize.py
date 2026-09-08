import pandas as pd
from lists import player_data_cat

def standardize():
    player_data = pd.read_csv("data/player_data.csv")

    num_cols = [col for col in player_data.columns if col not in player_data_cat]
    num_cols.remove('season_half_idx') 

    player_data[num_cols] = (
        player_data[num_cols] - player_data[num_cols].mean()
    ) / player_data[num_cols].std(ddof=0)

    transfer_data = pd.read_csv("data/transfer_data.csv")

    #standardize age_at_transfer, contract_left and market_val
    transfer_data[['age_at_transfer', 'market_val', 'contract_left_days']] = (transfer_data[['age_at_transfer', 'market_val', 'contract_left_days']] - transfer_data[['age_at_transfer', 'market_val', 'contract_left_days']].mean()) / transfer_data[['age_at_transfer', 'market_val', 'contract_left_days']].std(ddof=0)
    return player_data, transfer_data