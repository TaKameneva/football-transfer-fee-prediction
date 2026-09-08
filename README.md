# Football Transfer Fee Prediction

An end-to-end pipeline that predicts football transfer fees from pre-transfer player performance
data: scraping player statistics and transfer records, engineering features from each player's
form in the seasons before their move, and training/tuning gradient-boosted models (XGBoost,
CatBoost) alongside sklearn baselines to predict both the fee itself (regression) and a fee
bracket (classification).

## Pipeline

```
scraping            fbrefscraper.py / transfermarketscraper.py  →  raw per-league-season CSVs
                     (Selenium + BeautifulSoup)

data assembly        build_player_data_csv.py, build_transfer_data_csv.py
                     → data/player_data.csv, data/transfer_data.csv

standardization       standardize.py
                     z-score normalizes numeric player stats

feature engineering    dataprocessor.py
                     for each transfer, aggregates the player's stats over the N
                     season-halves before the move, encodes categoricals
                     (label or one-hot), and derives a fee-bracket classification
                     target alongside the log-transformed fee regression target

modeling                run.py / models.py
                     baseline sklearn models, then tuned XGBoost (regression +
                     classification) and CatBoost, evaluated and logged

reporting                 visualizations.py
                     feature-importance and performance charts from the run's
                     saved results
```

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

## Notes

This started as a university project (hence the course-code repo history); the scraping,
feature engineering, and tuned-hyperparameter choices reflect the scope of that assignment
rather than a production system.
