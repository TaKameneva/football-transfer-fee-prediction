import pandas as pd
import numpy as np
from lists import transfer_data_keep, league_map, transfer_columns_full
import sys

transfer_data1 = pd.read_csv('data/transfers_dataset_4.csv', names=transfer_columns_full)
transfer_data2 = pd.read_csv('data/transfers_dataset_5.csv', names=transfer_columns_full)

if transfer_data1.shape[1] != transfer_data2.shape[1]:
    print("ERROR: transfer_data1 and transfer_data2 have different number of columns.")
    sys.exit(1)

transfer_data = pd.concat([transfer_data1, transfer_data2], ignore_index=True)
print(f'Transfer data initial shape: {transfer_data.shape}')
#############################################
#create transfer_window_idx column
transfer_data['transfer_date'] = pd.to_datetime(transfer_data['transfer_date'], dayfirst=True)
month = transfer_data['transfer_date'].dt.month
year  = transfer_data['transfer_date'].dt.year
summer_mask = month.between(4, 9)
transfer_data['transfer_window_idx'] = np.where(summer_mask, (year - 2017) * 2, (year - 2017) * 2 + 1)
#get rid of all transfers before sumer 2017
transfer_data = transfer_data[(transfer_data['transfer_window_idx'] >= 0) & (transfer_data['transfer_window_idx'] <= 16)].reset_index(drop=True)

#get rid of all 'End of Loan' transfers
transfer_data = transfer_data[transfer_data['transfer_fee'] != 'End of loan'].reset_index(drop=True)

#replace 'Free Transfer' with 0
transfer_data['transfer_fee'] = transfer_data['transfer_fee'].replace('Free Transfer', '0')

#extract year of age of transfer
transfer_data["age_at_transfer"] = pd.to_numeric(transfer_data["age_at_transfer"].str[:2], errors="coerce")

#############################################
#convert contract_left to a number
contract_left_days = []
for length in transfer_data['contract_left']:
    if length != np.nan and isinstance(length, str):
        if 'Years' in length:
            years = length.split('Years')[0].strip()[-2:]
        else:
            years = 0
        if 'Months' in length:
            months = length.split('Months')[0].strip()[-2:]
        else:
            months = 0
        if 'Days' in length:
            days = length.split('Days')[0].strip()[-2:]
        else:
            days = 0
        total_days = int(years) * 365 + int(months) * 30 + int(days)
        contract_left_days.append(total_days)
    else:
        contract_left_days.append(np.nan)

if len(contract_left_days) == len(transfer_data):
    transfer_data['contract_left_days'] = contract_left_days

#############################################
#get rid of rows without fee
transfer_data = transfer_data.dropna(subset=['transfer_fee']).reset_index(drop=True)

#get rid of loans
transfer_data = transfer_data[transfer_data['is_loan'] == 0].reset_index(drop=True)

#only keep transfers from the 8 focused leagues
transfer_data['selling_league'] = transfer_data['selling_league'].map(league_map).fillna("OTHER")
transfer_data = transfer_data[transfer_data['selling_league'] != "OTHER"].reset_index(drop=True)

#only keep relevant columns
transfer_data = transfer_data[transfer_data_keep]

#############################################
#printing and saving
print('-'*50)
print("Final transfer data shape and NaN summary:")
print(transfer_data.shape)
print(transfer_data.isna().sum())
print('-'*50)

#############################################
#handle NaN values
transfer_data['market_val'] = transfer_data['market_val'].fillna(transfer_data['market_val'].median())
transfer_data['age_at_transfer'] = transfer_data['age_at_transfer'].fillna(transfer_data['age_at_transfer'].median())
transfer_data['contract_left_days'] = transfer_data['contract_left_days'].fillna(transfer_data['contract_left_days'].median())
print("[DataProcessor] Creating availability flags & fixing NaN ratios...")


transfer_data = transfer_data[pd.to_numeric(transfer_data['transfer_fee']) > 0].reset_index(drop=True)
#save to csv after handling NaN values
transfer_data.to_csv('data/transfer_data.csv', index=False)
print("After handling NaN values using median:")
print(transfer_data.shape)
print(f'Total number of NaN values: {transfer_data.isna().sum().sum()}')

for col in transfer_data.columns:
    print(f"{col}: {transfer_data[col].dtype}")
#############################################