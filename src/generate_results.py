"""
Generates the artifacts needed for a results write-up:

  results/main_performance_table.csv           regression models x encoders, test-set metrics
  results/classification_performance_table.csv classification models x encoders, test-set metrics
  results/dataset_summary.csv                  dataset size / split context
  results/fee_class_boundaries.csv             fee-bracket edges used for the classification target
  results/figures/predicted_vs_actual.png
  results/figures/model_comparison.png
  results/figures/feature_importance.png
  results/figures/confusion_matrix.png

Does not write any narrative/markdown -- only tables and figures. Reuses the
same DataProcessor pipeline and model definitions as run.py; nothing in
dataprocessor.py / models.py / run.py is modified.

Run from the project root:  python src/generate_results.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import xgboost as xgb
from catboost import CatBoostRegressor

from dataprocessor import DataProcessor
from standardize import load_data
from lists import cat_var, seasons
from models import get_all_models, get_all_classification_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = 1
POSITION = None

# Same tuned hyperparameters run.py uses (found via prior RandomizedSearchCV
# runs; kept here as constants rather than re-searching).
XGB_REG_PARAMS = {
    "label":  {'n_estimators': 700, 'learning_rate': 0.13210526315789473, 'min_child_weight': 2,
               'colsample_bytree': 1.0, 'max_depth': 4, 'subsample': 0.9, 'reg_alpha': 0, 'reg_lambda': 10},
    "onehot": {'n_estimators': 700, 'learning_rate': 0.10157894736842105, 'min_child_weight': 5,
               'colsample_bytree': 0.7, 'max_depth': 7, 'subsample': 0.5, 'reg_alpha': 0, 'reg_lambda': 5},
}
XGB_CLF_PARAMS = {
    "label":  {'n_estimators': 800, 'learning_rate': 0.27928571428571425, 'min_child_weight': 6,
               'colsample_bytree': 0.7, 'max_depth': 4, 'subsample': 0.7, 'reg_alpha': 0.01, 'reg_lambda': 5},
    "onehot": {'n_estimators': 800, 'learning_rate': 0.155, 'min_child_weight': 6,
               'colsample_bytree': 0.8, 'max_depth': 7, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1},
}
CATBOOST_PARAMS = {
    "label":  {'iterations': 1000, 'learning_rate': 0.2, 'depth': 8, 'subsample': 0.8, 'l2_leaf_reg': 20},
    "onehot": {'iterations': 1500, 'learning_rate': 0.09, 'depth': 9, 'subsample': 1.0, 'l2_leaf_reg': 20},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_regression_model(name, encoder, estimator, X_train, y_train, X_test, y_test):
    """Fits on log1p(fee), reports metrics both on the log scale (what the
    model was trained on) and converted back to euros (what a reader cares
    about): MAE_euro is the mean absolute error of the model's fee
    prediction in real euros, not of the log-transformed target."""
    log(f"  fitting {name} [{encoder}] ...")
    estimator.fit(X_train, y_train)
    pred_test_log = estimator.predict(X_test)

    actual_euro = np.expm1(y_test)
    pred_euro = np.clip(np.expm1(pred_test_log), a_min=0, a_max=None)

    record = {
        "model": name,
        "encoder": encoder,
        "MAE_euro": mean_absolute_error(actual_euro, pred_euro),
        "RMSE_euro": mean_squared_error(actual_euro, pred_euro) ** 0.5,
        "MAE_log": mean_absolute_error(y_test, pred_test_log),
        "RMSE_log": mean_squared_error(y_test, pred_test_log) ** 0.5,
        "R2_log": r2_score(y_test, pred_test_log),
        "_estimator": estimator,
        "_feature_names": list(X_test.columns),
        "_pred_euro": pred_euro,
        "_actual_euro": actual_euro.to_numpy(),
    }
    return record


def evaluate_classification_model(name, encoder, estimator, X_train, y_train, X_test, y_test):
    log(f"  fitting {name} [{encoder}] ...")
    estimator.fit(X_train, y_train)
    pred_test = estimator.predict(X_test)

    record = {
        "model": name,
        "encoder": encoder,
        "Accuracy": accuracy_score(y_test, pred_test),
        "Precision": precision_score(y_test, pred_test, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, pred_test, average='weighted', zero_division=0),
        "F1": f1_score(y_test, pred_test, average='weighted', zero_division=0),
        "_estimator": estimator,
        "_feature_names": list(X_test.columns),
        "_pred": pred_test,
        "_actual": y_test.to_numpy(),
    }
    return record


# ---------------------------------------------------------------------------
# Load data once, then run one DataProcessor per encoder
# ---------------------------------------------------------------------------

log("Loading player/transfer data ...")
full_player_data, transfer_data = load_data()

reg_records = []
clf_records = []
dataset_summary = {}
fee_class_bins = None

for encoder in ["label", "onehot"]:
    log(f"=== Building dataset for encoder={encoder} ===")
    processor = DataProcessor(full_player_data, transfer_data, TIMEFRAME, POSITION, encoder)

    train = processor.training_data.copy()
    test = processor.test_data.copy()
    for col in train.columns:
        train[col] = pd.to_numeric(train[col], errors='coerce')
    for col in test.columns:
        test[col] = pd.to_numeric(test[col], errors='coerce')

    X_train = train.drop(columns=['transfer_fee'])
    X_test = test.drop(columns=['transfer_fee'])
    y_train = np.log1p(pd.to_numeric(train['transfer_fee'], errors='coerce'))
    y_test = np.log1p(pd.to_numeric(test['transfer_fee'], errors='coerce'))

    train_class, test_class = processor.class_train_data, processor.class_test_data
    X_train_class = train_class.drop(columns=['fee_class', 'transfer_fee'])
    X_test_class = test_class.drop(columns=['fee_class', 'transfer_fee'])
    y_train_class = pd.to_numeric(train_class['fee_class'], errors='coerce')
    y_test_class = pd.to_numeric(test_class['fee_class'], errors='coerce')

    if fee_class_bins is None:
        # Bin edges are fit on the training split's transfer_fee before any
        # encoder-specific step runs, so they're identical across encoders.
        fee_class_bins = processor._fee_class_bins

    if encoder == "label":
        dataset_summary["Training samples"] = len(train)
        dataset_summary["Test samples"] = len(test)
        dataset_summary["Transfers used (train + test)"] = len(train) + len(test)
    dataset_summary[f"Features ({encoder} encoding)"] = X_train.shape[1]

    # -- baseline sklearn regression models (same ones run.py evaluates) --
    for m in get_all_models():
        rec = evaluate_regression_model(m.name, encoder, m.estimator, X_train, y_train, X_test, y_test)
        reg_records.append(rec)

    # -- tuned XGBoost regression --
    xgb_reg_tuned = xgb.XGBRegressor(**XGB_REG_PARAMS[encoder], random_state=42, n_jobs=-1,
                                      objective='reg:squarederror')
    reg_records.append(evaluate_regression_model(
        "XGBoost (tuned)", encoder, xgb_reg_tuned, X_train, y_train, X_test, y_test))

    # -- CatBoost fit on the *encoded* features (fair comparison against
    # XGBoost's encoder-dependent input), separate from CatBoost's native
    # categorical handling used below. --
    catboost_encoded = CatBoostRegressor(**CATBOOST_PARAMS[encoder], loss_function="RMSE",
                                          random_seed=42, silent=True)
    reg_records.append(evaluate_regression_model(
        "CatBoost (encoded features, tuned)", encoder, catboost_encoded, X_train, y_train, X_test, y_test))

    # -- baseline sklearn classification models --
    for m in get_all_classification_models():
        rec = evaluate_classification_model(m.name, encoder, m.estimator,
                                             X_train_class, y_train_class, X_test_class, y_test_class)
        clf_records.append(rec)

    # -- tuned XGBoost classifier --
    xgb_clf_tuned = xgb.XGBClassifier(**XGB_CLF_PARAMS[encoder], random_state=42, n_jobs=-1,
                                       objective='multi:softprob')
    clf_records.append(evaluate_classification_model(
        "XGBoostClassifier (tuned)", encoder, xgb_clf_tuned,
        X_train_class, y_train_class, X_test_class, y_test_class))

    if encoder == "label":
        # CatBoost's native categorical handling doesn't depend on the
        # --encoder choice at all (it uses the raw, unencoded columns), so
        # it only needs to run once.
        log("=== CatBoost with native categorical handling (encoder-independent) ===")
        train_cat = processor.catboost_train_data
        test_cat = processor.catboost_test_data
        X_train_cat = train_cat.drop(columns=['transfer_fee'])
        X_test_cat = test_cat.drop(columns=['transfer_fee'])
        y_train_cat = np.log1p(pd.to_numeric(train_cat['transfer_fee'], errors='coerce'))
        y_test_cat = np.log1p(pd.to_numeric(test_cat['transfer_fee'], errors='coerce'))
        cat_features = [c for c in cat_var if c in X_train_cat.columns]

        cat_baseline = CatBoostRegressor(random_seed=42, silent=True, cat_features=cat_features)
        reg_records.append(evaluate_regression_model(
            "CatBoost (native categorical, baseline)", "n/a", cat_baseline,
            X_train_cat, y_train_cat, X_test_cat, y_test_cat))

        cat_tuned = CatBoostRegressor(**CATBOOST_PARAMS["label"], loss_function="RMSE",
                                       random_seed=42, silent=True, cat_features=cat_features)
        reg_records.append(evaluate_regression_model(
            "CatBoost (native categorical, tuned)", "n/a", cat_tuned,
            X_train_cat, y_train_cat, X_test_cat, y_test_cat))

dataset_summary["Transfers in raw dataset"] = len(transfer_data)
dataset_summary["Distinct players in transfer records"] = transfer_data['player_name'].nunique()
dataset_summary["Distinct players in performance stats pool"] = full_player_data['player_name'].nunique()
dataset_summary["Seasons covered"] = f"{seasons[0]} to {seasons[-1]} ({len(seasons)} seasons)"
dataset_summary["Split method"] = ("Temporal split by transfer_window_idx (no shuffling): every test "
                                    "transfer occurred strictly after every training transfer.")

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

log("Writing tables ...")

reg_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in reg_records])
reg_df = reg_df.sort_values("MAE_euro").reset_index(drop=True)
reg_df.to_csv(RESULTS_DIR / "main_performance_table.csv", index=False)

clf_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in clf_records])
clf_df = clf_df.sort_values("F1", ascending=False).reset_index(drop=True)
clf_df.to_csv(RESULTS_DIR / "classification_performance_table.csv", index=False)

pd.DataFrame(list(dataset_summary.items()), columns=["Property", "Value"]).to_csv(
    RESULTS_DIR / "dataset_summary.csv", index=False)

bins = np.asarray(fee_class_bins, dtype=float)
bounds_df = pd.DataFrame({
    "class": list(range(len(bins) - 1)),
    "lower_bound_eur": bins[:-1],
    "upper_bound_eur": bins[1:],
})
bounds_df.to_csv(RESULTS_DIR / "fee_class_boundaries.csv", index=False)

print("\n=== MAIN PERFORMANCE TABLE (regression, sorted by MAE_euro) ===")
print(reg_df.drop(columns=[]).to_string(index=False))
print("\n=== CLASSIFICATION PERFORMANCE TABLE (sorted by F1) ===")
print(clf_df.to_string(index=False))
print("\n=== DATASET SUMMARY ===")
for k, v in dataset_summary.items():
    print(f"{k}: {v}")
print("\n=== FEE CLASS BOUNDARIES ===")
print(bounds_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure 1: predicted vs actual (best regression model by MAE_euro)
# ---------------------------------------------------------------------------

log("Plotting predicted_vs_actual.png ...")
best_reg = min(reg_records, key=lambda r: r["MAE_euro"])
actual = best_reg["_actual_euro"]
pred = best_reg["_pred_euro"]

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(actual, pred, alpha=0.5, s=18, edgecolor='none')
lo = max(1.0, min(actual.min(), pred.min()))
hi = max(actual.max(), pred.max())
ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xscale('symlog', linthresh=1e5)
ax.set_yscale('symlog', linthresh=1e5)
ax.set_xlabel('Actual transfer fee (€)')
ax.set_ylabel('Predicted transfer fee (€)')
ax.set_title(f'Predicted vs Actual Transfer Fee\n{best_reg["model"]} [{best_reg["encoder"]}]')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: model comparison (CatBoost vs XGBoost, label vs onehot)
# ---------------------------------------------------------------------------

log("Plotting model_comparison.png ...")
comparison_labels = [
    ("CatBoost (encoded features, tuned)", "label", "CatBoost + label"),
    ("CatBoost (encoded features, tuned)", "onehot", "CatBoost + one-hot"),
    ("XGBoost (tuned)", "label", "XGBoost + label"),
    ("XGBoost (tuned)", "onehot", "XGBoost + one-hot"),
]
comp_values = []
comp_names = []
for model_name, enc, disp in comparison_labels:
    match = reg_df[(reg_df["model"] == model_name) & (reg_df["encoder"] == enc)]
    if not match.empty:
        comp_values.append(match["MAE_euro"].iloc[0] / 1e6)
        comp_names.append(disp)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(comp_names, comp_values, color=['#4C72B0', '#4C72B0', '#DD8452', '#DD8452'])
ax.set_ylabel('Test MAE (€ millions)')
ax.set_title('Model Comparison: Encoder Choice vs Test MAE')
for bar, val in zip(bars, comp_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
            ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: feature importance (best regression model)
# ---------------------------------------------------------------------------

log("Plotting feature_importance.png ...")
best_estimator = best_reg["_estimator"]
feature_names = best_reg["_feature_names"]

if hasattr(best_estimator, "feature_importances_"):
    importances = pd.Series(best_estimator.feature_importances_, index=feature_names)
elif hasattr(best_estimator, "coef_"):
    importances = pd.Series(np.abs(best_estimator.coef_), index=feature_names)
else:
    importances = None

if importances is not None:
    top = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index[::-1], top.values[::-1], color='#4C72B0')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {len(top)} Features — {best_reg["model"]} [{best_reg["encoder"]}]')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
else:
    log(f"  {best_reg['model']} exposes neither feature_importances_ nor coef_, skipping.")

# ---------------------------------------------------------------------------
# Figure 4: confusion matrix (best classification model by F1)
# ---------------------------------------------------------------------------

log("Plotting confusion_matrix.png ...")
best_clf = max(clf_records, key=lambda r: r["F1"])
cm = confusion_matrix(best_clf["_actual"], best_clf["_pred"])
class_labels = [f"Class {i}" for i in range(cm.shape[0])]

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(class_labels)))
ax.set_yticks(range(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — {best_clf["model"]} [{best_clf["encoder"]}]')
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

log("Done. Tables in results/, figures in results/figures/.")
