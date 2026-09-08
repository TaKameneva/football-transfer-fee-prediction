from lists import leagues, seasons, player_data_drop
import pandas as pd
import numpy as np
import os


player_data = pd.DataFrame()

for league in leagues:
    idx = -2017
    corr = 0
    for season in seasons:
        index_column = []
        path = f"scraped_seasons/{league}_{season}.csv"

        if not os.path.exists(path):
            corr += 1
            continue

        df = pd.read_csv(path)
        half = (len(df)//2)
        index_column.extend([float(idx + int(season[:4]) + corr)] * half)
        index_column.extend([float(idx + int(season[:4]) + 1) + corr] * (len(df) - half))
        if len(df) == len(index_column):
            df['season_half_idx'] = index_column
        else:
            print('Length does not match')
            print(f'Df: {len(df)}, List: {len(index_column)}')
            break
        player_data = pd.concat([player_data, df], ignore_index=True)
        corr += 1

#get rid of rows with missing league (goalkeeper)
player_data = player_data.dropna(subset=['league', 'player_name'])

#only keep columns relevant for analysis
player_data = player_data.drop(columns=player_data_drop)

# NOTE: numeric columns are intentionally left unscaled here. Standardizing
# at this stage would fit mean/std over the full dataset, before it's ever
# split into train/test -- leaking test-set statistics into training.
# Scaling now happens in DataProcessor.fit_scaler(), fit on the training
# split only (see src/dataprocessor.py).
print(f'Assembled. Shape: {player_data.shape}')

###############################################
#handle nan
print("NaN summary per column:")

for col in player_data.columns:
    nan_mask = player_data[col].isna()
    if nan_mask.any():
        first_idx = nan_mask.idxmax()           # first NaN index
        count = nan_mask.sum()                  # total NaNs
        print(f"{col}: first NaN at row {first_idx}, total {count}")
player_data['minutes_played'] = player_data['minutes_played'].fillna(0)

perc_cols = [col for col in player_data.columns if col.endswith('_perc')]
player_data[perc_cols] = player_data[perc_cols].fillna(player_data[perc_cols].median())

#Save to CSV
print('to csv...')
player_data.to_csv("data/player_data.csv", index=False)
print(f'Saved in csv. Final Shape: {player_data.shape}')

for col in player_data.columns:
    print(f"{col}: {player_data[col].dtype}")