# Football Transfer Fee Prediction

An end-to-end pipeline that predicts football transfer fees from pre-transfer player performance
data: scraping player statistics and transfer records, engineering features from each player's
form in the seasons before their move, and training/tuning gradient-boosted models (XGBoost,
CatBoost) alongside sklearn baselines to predict both the fee itself (regression) and a fee
bracket (classification).

## Pipeline

```
scraping            fbrefscraper.py / transfermarketscraper.py  →  raw per-league-season CSVs
                     (Selenium + BeautifulSoup), 8 leagues x 8 seasons (2017-18 to 2024-25):
                     Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Primeira Liga,
                     Eredivisie, Championship

data assembly        build_player_data_csv.py, build_transfer_data_csv.py
                     → data/player_data.csv, data/transfer_data.csv (raw, unscaled)
                     662,867 player-season-half stat rows (11,108 players) and
                     3,762 transfer records

feature engineering    dataprocessor.py
                     for each transfer, aggregates the player's pre-transfer stats
                     over the N season-halves before the move, splits chronologically
                     into train/test, then fits correlation filtering, categorical
                     encoding, fee-bracket boundaries, and feature scaling on the
                     training split only and applies them to test (see below)

modeling                run.py / models.py / generate_results.py
                     sklearn baselines (Linear/Logistic Regression, Random Forest,
                     Gradient Boosting) plus tuned XGBoost and CatBoost
                     (RandomizedSearchCV), compared across label vs. one-hot
                     encoding and CatBoost's native categorical handling —
                     8+ model/encoder combinations evaluated for regression alone

reporting                 visualizations.py, generate_results.py
                     feature-importance and performance charts from the run's
                     saved results
```

## Results

Evaluated on a temporally held-out test set (531 transfers from later transfer windows than
anything the models trained or were tuned on — see [Train/test methodology](#traintest-methodology)).
Full sweep across every model/encoder combination in
[`src/generate_results.py`](src/generate_results.py); tables in [`results/`](results).

**Regression — predicting the exact fee:**
- Best model: GradientBoosting (one-hot) — **MAE €5.50M**, R² (log-fee) **0.80** on unseen future
  transfers.
- That's a **65% error reduction** vs. the naive baseline of always guessing the training-set
  average fee (€15.7M MAE) — the model is clearly extracting real signal from pre-transfer
  performance stats, not fitting noise.

**Classification — predicting the fee bracket** (5 quantile classes, see
[`results/fee_class_boundaries.csv`](results/fee_class_boundaries.csv)):
- Best model: GradientBoostingClassifier (label) — **59% accuracy / F1 0.59**, vs. 20% for random
  guessing on 5 balanced classes — **~3x better than chance**.
- The confusion matrix below shows most mistakes land on the *neighboring* bracket, not the
  opposite end of the market: the model essentially never mistakes a bargain-tier player for a
  world-class one.

| ![Predicted vs actual transfer fee](results/figures/predicted_vs_actual.png) | ![Model comparison across encoders](results/figures/model_comparison.png) |
|---|---|

![Confusion matrix for fee-bracket classification](results/figures/confusion_matrix.png)

**Limitations**
- €5.50M MAE is still ~40% of the median test fee (€13.9M) in absolute terms — transfer fees are
  shaped by negotiation dynamics, release clauses, and bidding-war hype that pre-transfer
  performance stats alone can't capture, so exact-euro prediction has a hard ceiling regardless
  of model choice.
- Hyperparameter-tuned XGBoost and CatBoost *underperformed* the untuned GradientBoosting
  baseline on the held-out test set — a sign the tuning search overfit its own validation split
  rather than the true temporal test distribution, worth revisiting with tighter CV.
- ~2,655 usable transfers after position/history filtering is a modest sample against up to 242
  features (one-hot encoding), which caps how much signal boosted trees can extract before
  overfitting becomes the binding constraint.

## Setup

```bash
pip install -r requirements.txt
```

Scraping additionally requires a local Chrome installation (Selenium drives a real browser).

## Usage

Run all commands from the repository root.

**1. Rebuild the datasets** (optional — `data/` already contains the processed CSVs from a
prior run; the raw per-league-season scrape files are not included due to size, so
`build_player_data_csv.py` can only be re-run against your own freshly scraped data):

```bash
python src/build_player_data_csv.py
python src/build_transfer_data_csv.py
```

**2. Train and evaluate models:**

```bash
python src/run.py --encoder label
# or
python src/run.py --encoder onehot --timeframe 2 --position forward
```

| Flag | Description |
|---|---|
| `--position` | restrict to one position group: `forward`, `winger`, `midfielder`, `defender`, `goalkeeper` |
| `--timeframe` | number of season-halves of pre-transfer form to aggregate per player (1–4) |
| `--encoder` | categorical encoding strategy: `label` or `onehot` |

This trains baseline sklearn regressors/classifiers, then tuned XGBoost and CatBoost models,
and writes:
- `data/train_data.csv`, `data/test_data.csv` — the processed train/test split
- `results/*.csv` — feature importances for each tuned model
- `logs/model_run_output_<encoder>.log` — full run log with metrics

**3. Generate report figures** from a completed run's outputs:

```bash
python src/visualizations.py
```

## Data

`data/` ships with the already-processed datasets (`player_data.csv` inside the zip,
`transfer_data.csv`, and the train/test splits) so the modeling step works out of the box.
The raw per-league-season scrape files that feed `build_player_data_csv.py` are not included —
regenerating them requires re-running the scrapers.

> **Known limitation:** `data/player_data.csv.zip` was originally built by an earlier version
> of `build_player_data_csv.py` that standardized numeric player stats over the *entire*
> dataset before any split existed. That baked-in scaling can't be undone without the raw
> per-league-season scrape files, which aren't included. `build_player_data_csv.py` itself no
> longer does this — regenerating the dataset from a fresh scrape would produce genuinely raw,
> unscaled stats, consistent with how the rest of the pipeline now works.

## Train/test methodology

The original pipeline shuffled a random 80/20 split *after* fitting standardization,
correlation-based feature filtering, categorical encoding, and fee-bracket boundaries on the
full dataset — leaking test-set statistics into every one of those steps. It also standardized
player stats twice (once in `build_player_data_csv.py`, again in the loader), silently
compounding the leak.

The pipeline now:
1. **Splits first, chronologically.** `DataProcessor.temporal_split()` sorts transfers by
   `transfer_window_idx` and takes the most recent slice as the test set — no shuffling. This
   also better reflects the real use case: predicting *future* transfer fees from *past* data,
   rather than interpolating within a randomly shuffled dataset.
2. **Fits every transform on the training split only** — correlation-based feature filtering
   (`fit_correlation_filter`), categorical encoding (`fit_categorical_encoder`, with unseen
   test-only categories mapped to an explicit "unknown" bucket), fee-bracket quantile edges
   (`fit_fee_class`, with test fees outside the train range clamped to the nearest bucket
   instead of becoming `NaN`), and feature scaling (`fit_scaler`).
3. **Applies those fitted transforms to test unchanged** — test data never influences which
   columns get dropped, how categories get encoded, where class boundaries fall, or the
   scaler's mean/std.

CatBoost's inputs (`catboost_train_data` / `catboost_test_data`) stay raw and unscaled by
design — CatBoost handles native categoricals and doesn't need scaled numeric features.

## Notes

This started as a university project (hence the course-code repo history); the scraping,
feature engineering, and tuned-hyperparameter choices reflect the scope of that assignment
rather than a production system.
