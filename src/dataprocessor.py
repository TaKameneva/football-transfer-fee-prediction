import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
import numpy as np
from lists import cat_var, base_columns
import sys

# Position map
position_map = {
    'FW': 'forward', 'ST': 'forward', 'CF': 'forward',
    'LW': 'winger', 'RW': 'winger', 'LM': 'winger', 'RM': 'winger',
    'RB': 'defender', 'LB': 'defender', 'RWB': 'defender', 'LWB': 'defender', 'CB': 'defender',
    'CM': 'midfielder', 'DM': 'midfielder', 'AM': 'midfielder',
    'GK': 'goalkeeper'
}
def threshold_filter(data, threshold, exclude_cols):
    if isinstance(exclude_cols, (str, tuple, set)):
        exclude_cols = list(exclude_cols)
    elif isinstance(exclude_cols, list):
        exclude_cols = [col for col in exclude_cols]
    print(f"Excluding columns from correlation check: {exclude_cols}")
    independent_df = data.drop(columns=exclude_cols, errors='ignore')
    corr_matrix = independent_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    dropped = []
    for col in upper.columns:
        if any(upper[col] > threshold):
            dropped.append(col)
    df_clean = data.drop(columns=dropped)  
    return df_clean   
class DataProcessor:
    def __init__(self, full_player_data, transfer_data, timeframe, position=None, encoder='label'):
        self._data = pd.DataFrame()
        self._training_data = pd.DataFrame()
        self._test_data = pd.DataFrame()
        self._timeframe = timeframe
        self._position = position
        self._transfer_data = transfer_data
        self._encoder = encoder
        self._player_groups = dict(tuple(full_player_data.groupby('player_name')))
        self.loop_transfers()

    def find_main_position(self, position_str):
        try:
            abbrevs = [p.strip() for p in position_str.split(',') if p.strip()]
            mapped = [position_map.get(p) for p in abbrevs if position_map.get(p)]
            if not mapped:
                return None
            return Counter(mapped).most_common(1)[0][0]
        except:
            return None

    def aggregate_data(self, player_data, relevant_season_halves):
        slices_list = []
        for s in relevant_season_halves:
            df_slice = player_data[player_data['season_half_idx'] == s]
            if not df_slice.empty:
                slices_list.append(df_slice)
        if not slices_list:
            return pd.DataFrame()

        slices = pd.concat(slices_list, ignore_index=True)
        agg_dict = {}
        for column in slices.columns:
            if column.endswith('perc'):
                values = round(slices[column].mean(), 2)
            else:
                try:
                    numeric_vals = pd.to_numeric(slices[column])
                    values = round(numeric_vals.sum(), 2)
                except:
                    values = ', '.join(map(str, slices[column].dropna().unique().tolist()))
            agg_dict[column] = values

        player_data_agg = pd.DataFrame([agg_dict])
        return player_data_agg

    def append_data(self, single_transfer_data, player_data):
        full_row = pd.concat([single_transfer_data.reset_index(drop=True), player_data.reset_index(drop=True)], axis=1)
        self._data = pd.concat([self._data, full_row], ignore_index=True)

    def prepare_data(self):
        # Derive main_position
        self._data['position'] = self._data['position'].apply(self.find_main_position)
        self._data = self._data.dropna(subset=['position']).reset_index(drop=True)
        # Ratio engineering
        self._data['shot_creation_ratio'] = round(
            self._data['shot_creating_actions'] /
            (self._data['carries_into_final_3rd'] +
             self._data['passes_final_3rd'] +
             self._data['touches_att_3rd']).replace(0, np.nan), 4
        )

        # Process ratio columns
        ratio_cols = [
            "aerials_won_perc", "take_ons_success_perc",
            "tackeled_during_take_on_perc", "dribbles_tackeled_perc",
            "pass_compl_perc", "short_pass_compl_perc",
            "medium_pass_compl_perc", "long_pass_compl_perc",
            "shot_creation_ratio"
        ]

        print("[DataProcessor] Creating availability flags & fixing NaN ratios...")

        for col in ratio_cols:
            flag = col + "_available"
            self._data[flag] = self._data[col].notna().astype(int)
            self._data[col] = self._data[col].fillna(0)
        
        self._data.drop(columns=['player_name', 'season_half_idx'], inplace=True, errors='ignore')
        self._data = threshold_filter(self._data, 0.8, ['transfer_fee'] + cat_var)        
            
    def train_test_splitter(self):
        self._training_data, self._test_data = train_test_split(
            self._data, test_size=0.2, shuffle=True, random_state=42
        )

    def loop_transfers(self):
        print(f'Number of Transfers: {len(self._transfer_data)}')
        print('Looping Transfers...')

        for i, single_transfer_data in self._transfer_data.iterrows():

            single_transfer_data = single_transfer_data.to_frame().T
            player_name = single_transfer_data['player_name'].item()

            if player_name not in self._player_groups:
                continue

            player_data_extended = self._player_groups[player_name]

            # Filter by main position 
            if self._position is not None:
                raw_position = player_data_extended["position"].iloc[0]
                player_main_pos = self.find_main_position(raw_position)
                if player_main_pos != self._position:
                    continue

            # Determine relevant halves
            transfer_window_idx = single_transfer_data['transfer_window_idx'].values[0]
            relevant_season_halves = [
                i for i in range(int(transfer_window_idx) - self._timeframe, int(transfer_window_idx))
                if i >= 0
            ]

            # Aggregate data
            player_data_agg = self.aggregate_data(player_data_extended, relevant_season_halves)

            if player_data_agg.empty:
                continue
            self.append_data(single_transfer_data, player_data_agg)

        self.prepare_data()
        self._data_raw = self._data.copy()
        self.encode_cat_vars()
        self.add_fee_class()
        self.train_test_splitter()

        # Create matching splits for raw data
        self._training_data_raw = self._data_raw.loc[self._training_data.index].copy()
        self._test_data_raw = self._data_raw.loc[self._test_data.index].copy()
        
        self._training_data_class = self._data_class.loc[self._training_data.index].copy()
        self._test_data_class = self._data_class.loc[self._test_data.index].copy()

        
    def add_fee_class(self):

        q = 5
        self._data_class['fee_class'] = pd.qcut(
            self._data_class['transfer_fee'],
            q=q,
            labels=[i for i in range(q)], 
            duplicates='drop'
        )
        print(f"[DataProcessor] Fee class distribution:\n{self._data_class['fee_class'].value_counts().sort_index()}")
        print("[DataProcessor] Added fee_class column for classification.")

    def encode_cat_vars(self):
        if self._encoder == 'label':
            print(self._data[cat_var].head())
            for col in cat_var:
                
                print(f"[DataProcessor] Label encoding column: {col}")
                print(self._data[col].dtype, self._data[col].head())
                le = LabelEncoder()
                self._data[col] = le.fit_transform(self._data[col].astype(str)).squeeze()
        elif self._encoder == 'onehot':
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            ohe_df = pd.DataFrame(ohe.fit_transform(self._data[cat_var].astype(str)), 
                                columns=ohe.get_feature_names_out(cat_var),
                                index=self._data.index)
            self._data = pd.concat([self._data.drop(cat_var, axis=1), ohe_df], axis=1)
        else:
            raise ValueError("Mode should be either 'label' or 'onehot'")
        
        self._data_class = self._data.copy()
        self._data_class = self._data_class.drop(columns=['player_name', 'season_half_idx'], errors='ignore')
        for col in self._data_class.columns:
            if self._data_class[col].dtype == "object":
                self._data_class[col] = pd.to_numeric(self._data_class[col], errors="coerce")

        self._data_class = self._data_class.fillna(0)
    # -------------------------------------------------------------------
    @property
    def training_data(self):
        return self._training_data

    @property
    def test_data(self):
        return self._test_data

    # CatBoost access (raw categoricals)
    @property
    def catboost_train_data(self):
        return self._training_data_raw
    @property
    def catboost_test_data(self):
        return self._test_data_raw
    
    @property
    def class_train_data(self):
        return self._training_data_class
    @property
    def class_test_data(self):
        return self._test_data_class
