import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from collections import Counter
from lists import cat_var, base_columns

# ---------------------------------------------------------------------------
# Every fitted step below (correlation filter, categorical encoder, fee-class
# bin edges, numeric scaler) is fit on the training split only and then
# applied to test. The split itself is temporal (by transfer_window_idx, no
# shuffling), so test transfers are strictly later than every train transfer.
# This avoids leaking test-set statistics into preprocessing/training, which
# a random pre-split shuffle + globally-fit transforms would otherwise do.
# ---------------------------------------------------------------------------

# Position map
position_map = {
    'FW': 'forward', 'ST': 'forward', 'CF': 'forward',
    'LW': 'winger', 'RW': 'winger', 'LM': 'winger', 'RM': 'winger',
    'RB': 'defender', 'LB': 'defender', 'RWB': 'defender', 'LWB': 'defender', 'CB': 'defender',
    'CM': 'midfielder', 'DM': 'midfielder', 'AM': 'midfielder',
    'GK': 'goalkeeper'
}


def compute_correlated_columns(data, threshold, exclude_cols):
    """Return columns whose absolute correlation with another column exceeds
    `threshold`, computed on `data` alone (pass the training split only)."""
    exclude_cols = list(exclude_cols)
    print(f"Excluding columns from correlation check: {exclude_cols}")
    independent_df = data.drop(columns=exclude_cols, errors='ignore')
    corr_matrix = independent_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    dropped = [col for col in upper.columns if any(upper[col] > threshold)]
    return dropped


class DataProcessor:
    def __init__(self, full_player_data, transfer_data, timeframe, position=None, encoder='label', test_size=0.2):
        self._data = pd.DataFrame()
        self._training_data = pd.DataFrame()
        self._test_data = pd.DataFrame()
        self._timeframe = timeframe
        self._position = position
        self._transfer_data = transfer_data
        self._encoder = encoder
        self._test_size = test_size
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
        # loop_transfers() builds each row from DataFrame.iterrows(), which
        # upcasts every value in a row to a single common dtype -- so numeric
        # columns (market_val, age_at_transfer, contract_left_days, ...)
        # arrive here as `object` even though their values are numeric. Coerce
        # them back before anything downstream (correlation filter, scaler)
        # runs, or those steps silently skip columns that look non-numeric.
        non_numeric_cols = set(cat_var) | {'player_name'}
        for col in self._data.columns:
            if col not in non_numeric_cols:
                self._data[col] = pd.to_numeric(self._data[col], errors='coerce')

        # Row-wise feature engineering: derives features from each row's own
        # values only, so it's safe to run before the split.
        self._data['position'] = self._data['position'].apply(self.find_main_position)
        self._data = self._data.dropna(subset=['position']).reset_index(drop=True)

        self._data['shot_creation_ratio'] = round(
            self._data['shot_creating_actions'] /
            (self._data['carries_into_final_3rd'] +
             self._data['passes_final_3rd'] +
             self._data['touches_att_3rd']).replace(0, np.nan), 4
        )

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

    def temporal_split(self):
        """Split by transfer_window_idx (no shuffling) so every test transfer
        happened strictly after every train transfer, mirroring how the model
        would actually be used: trained on the past, evaluated on the future."""
        self._data = self._data.sort_values('transfer_window_idx', kind='stable').reset_index(drop=True)
        split_idx = int(len(self._data) * (1 - self._test_size))
        self._training_data = self._data.iloc[:split_idx].copy()
        self._test_data = self._data.iloc[split_idx:].copy()
        print(
            f"[DataProcessor] Temporal split: {len(self._training_data)} train / "
            f"{len(self._test_data)} test rows. Train transfer_window_idx up to "
            f"{self._training_data['transfer_window_idx'].max()}, test from "
            f"{self._test_data['transfer_window_idx'].min()}."
        )

    def fit_correlation_filter(self, threshold=0.8):
        dropped = compute_correlated_columns(self._training_data, threshold, ['transfer_fee'] + cat_var)
        print(f"[DataProcessor] Dropping {len(dropped)} highly correlated columns "
              f"(threshold={threshold}, fit on train only): {dropped}")
        self._training_data = self._training_data.drop(columns=dropped, errors='ignore')
        self._test_data = self._test_data.drop(columns=dropped, errors='ignore')

    def fit_categorical_encoder(self):
        if self._encoder == 'label':
            self._label_encoders = {}
            for col in cat_var:
                train_categories = sorted(self._training_data[col].astype(str).unique().tolist())
                if 'unknown' not in train_categories:
                    train_categories.append('unknown')
                le = LabelEncoder()
                le.fit(train_categories)

                self._training_data[col] = le.transform(self._training_data[col].astype(str))

                # Categories seen only in test (e.g. a small club with no train
                # transfers) fall back to the "unknown" bucket instead of
                # crashing LabelEncoder.transform on an unseen label.
                test_vals = self._test_data[col].astype(str)
                test_vals = test_vals.where(test_vals.isin(le.classes_), other='unknown')
                self._test_data[col] = le.transform(test_vals)

                self._label_encoders[col] = le
            self._categorical_output_cols = list(cat_var)

        elif self._encoder == 'onehot':
            ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            train_ohe = ohe.fit_transform(self._training_data[cat_var].astype(str))
            test_ohe = ohe.transform(self._test_data[cat_var].astype(str))
            ohe_cols = list(ohe.get_feature_names_out(cat_var))

            train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_cols, index=self._training_data.index)
            test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_cols, index=self._test_data.index)

            self._training_data = pd.concat([self._training_data.drop(columns=cat_var), train_ohe_df], axis=1)
            self._test_data = pd.concat([self._test_data.drop(columns=cat_var), test_ohe_df], axis=1)

            self._onehot_encoder = ohe
            self._categorical_output_cols = ohe_cols
        else:
            raise ValueError("Mode should be either 'label' or 'onehot'")

        print(f"[DataProcessor] Encoded categorical columns ({self._encoder}), fit on train only.")

    def fit_scaler(self):
        """Standardize numeric features using mean/std computed on train only,
        then apply that same transform to test. Categorical (encoded) columns
        and availability flags are left alone; transfer_fee stays in raw
        units since run.py applies its own log1p transform to the target."""
        exclude = set(self._categorical_output_cols) | {'transfer_fee', 'transfer_window_idx'}
        exclude |= {c for c in self._training_data.columns if c.endswith('_available')}

        numeric_cols = [
            c for c in self._training_data.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(self._training_data[c])
        ]

        scaler = StandardScaler()
        self._training_data[numeric_cols] = scaler.fit_transform(self._training_data[numeric_cols])
        self._test_data[numeric_cols] = scaler.transform(self._test_data[numeric_cols])

        self._scaler = scaler
        self._scaled_columns = numeric_cols
        print(f"[DataProcessor] Scaled {len(numeric_cols)} numeric columns (fit on train only).")

    def fit_fee_class(self, q=5):
        _, bins = pd.qcut(
            self._training_data_class['transfer_fee'], q=q, retbins=True, duplicates='drop'
        )
        bins = bins.astype(float)
        bins[0], bins[-1] = -np.inf, np.inf
        labels = list(range(len(bins) - 1))

        self._training_data_class['fee_class'] = pd.cut(
            self._training_data_class['transfer_fee'], bins=bins, labels=labels, include_lowest=True
        ).astype(int)
        # Test fees are bucketed with the SAME train-derived bin edges; -inf/+inf
        # outer edges mean a test fee outside the train range still lands in the
        # nearest bucket instead of becoming NaN.
        self._test_data_class['fee_class'] = pd.cut(
            self._test_data_class['transfer_fee'], bins=bins, labels=labels, include_lowest=True
        ).astype(int)

        self._fee_class_bins = bins
        print(f"[DataProcessor] fee_class bin edges (fit on train only): {bins}")
        print(f"[DataProcessor] Train fee_class distribution:\n"
              f"{self._training_data_class['fee_class'].value_counts().sort_index()}")
        print(f"[DataProcessor] Test fee_class distribution:\n"
              f"{self._test_data_class['fee_class'].value_counts().sort_index()}")

    def _prepare_class_view(self, df):
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == "object":
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.fillna(0)

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
        self.temporal_split()
        self.fit_correlation_filter()

        # Raw (unscaled, unencoded-categorical) views for CatBoost, which
        # handles native categoricals and doesn't need scaled numeric input.
        self._training_data_raw = self._training_data.copy()
        self._test_data_raw = self._test_data.copy()

        self.fit_categorical_encoder()
        self.fit_scaler()

        # Classification-target views, derived from the already
        # encoded+scaled data so they share the same feature values.
        self._training_data_class = self._prepare_class_view(self._training_data)
        self._test_data_class = self._prepare_class_view(self._test_data)
        self.fit_fee_class()

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
