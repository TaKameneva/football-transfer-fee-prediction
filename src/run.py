import argparse
import os
import pandas as pd
import sys
import numpy as np

from dataprocessor import DataProcessor
from models import (
    get_all_models,
    get_all_classification_models,
    evaluate_classification,
    ModelRunner,
    tune_xgboost,
    tune_xgboost_classifier,
    tune_catboost,
    run_baseline_catboost
)

from standardize import load_data

from lists import cat_var, transfer_data_num
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--position", type=str, default=None, choices=['forward', 'winger', 'midfielder', 'defender', 'goalkeeper'])
parser.add_argument("--timeframe", type=int, default=1, choices=[1, 2, 3, 4])
parser.add_argument("--encoder", type=str, default="label", choices=["label", "onehot"])
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
log_filename = f"logs/model_run_output_{args.encoder}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# mode='w': each run overwrites its own log rather than appending
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

################################################
logger.info("[START] Loading data ...")
full_player_data, transfer_data = load_data()
logger.info(f'Player data: {full_player_data.shape}')
logger.info(f'Transfer data: {transfer_data.shape}')
################################################
logger.info("[START] Loading data using DataProcessor ...")

processor = DataProcessor(full_player_data, transfer_data, args.timeframe, args.position, args.encoder)
train = processor.training_data
test  = processor.test_data
for col in train.columns:
    train[col] = pd.to_numeric(train[col], errors='coerce')

for col in test.columns:
    test[col] = pd.to_numeric(test[col], errors='coerce')


train.to_csv("data/train_data.csv", index=False)
test.to_csv("data/test_data.csv", index=False)
X_train = train.drop(columns=['transfer_fee'])
X_test  = test.drop(columns=['transfer_fee'])

y_train = np.log1p(pd.to_numeric(train['transfer_fee'], errors='coerce'))
y_test  = np.log1p(pd.to_numeric(test['transfer_fee'], errors='coerce'))

logger.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')

################################################
logger.info("[INFO] Running baseline sklearn models ...")
models = get_all_models()
models_class = get_all_classification_models()
if args.encoder == "label":
    xgb_tuned_classification_params  = {'n_estimators': 800, 'learning_rate': 0.27928571428571425, 'min_child_weight': 6, 'colsample_bytree': 0.7, 'max_depth': 4, 'subsample': 0.7, 'reg_alpha': 0.01, 'reg_lambda': 5}
    xgb_tuned_regressiom_params = {'n_estimators': 700, 'learning_rate': 0.13210526315789473, 'min_child_weight': 2, 'colsample_bytree': 1.0, 'max_depth': 4, 'subsample': 0.9, 'reg_alpha': 0, 'reg_lambda': 10}
    catboost_tuned_params = {'iterations': 1000, 'learning_rate': 0.2, 'depth': 8, 'subsample': 0.8, 'l2_leaf_reg': 20}
elif args.encoder == "onehot":
    xgb_tuned_classification_params  ={'n_estimators': 800, 'learning_rate': 0.155, 'min_child_weight': 6, 'colsample_bytree': 0.8, 'max_depth': 7, 'subsample': 0.8, 'reg_alpha': 0, 'reg_lambda': 1}
    xgb_tuned_regressiom_params = {'n_estimators': 700, 'learning_rate': 0.10157894736842105, 'min_child_weight': 5, 'colsample_bytree': 0.7, 'max_depth': 7, 'subsample': 0.5, 'reg_alpha': 0, 'reg_lambda': 5}
    catboost_tuned_params = {'iterations': 1500, 'learning_rate': 0.09, 'depth': 9, 'subsample': 1.0, 'l2_leaf_reg': 20}

runner = ModelRunner(models=models)
runner_class = ModelRunner(models=models_class, evaluator=evaluate_classification)
train_class, test_class = processor.class_train_data, processor.class_test_data

X_train_class = train_class.drop(columns=['fee_class', 'transfer_fee'])
X_test_class  = test_class.drop(columns=['fee_class', 'transfer_fee'])

y_train_class = pd.to_numeric(train_class['fee_class'], errors='coerce')
y_test_class  = pd.to_numeric(test_class['fee_class'], errors='coerce')
results_class = runner_class.run(X_train_class, y_train_class, X_test_class, y_test_class)
logger.info("\n[INFO] Classification Model Performance:")
logger.info("\n" + results_class.to_string())
runner_class.print_results(results_class)
logger.info("\n[INFO] Regression Model Performance:")
df_results = runner.run(X_train, y_train, X_test, y_test)
runner.print_results(df_results)
logger.info("\n" + df_results.to_string())
logger.info("[DONE] Baseline sklearn results computed.\n")

logger.info("\n[INFO] Tuning XGBoost Classifier model ...")
best_gb_model, gb_tuned_metrics = tune_xgboost_classifier(X_train_class, y_train_class, X_test_class, y_test_class, xgb_tuned_classification_params)
logger.info("[INFO] XGBoost tuning completed.\n")

logger.info("\nModel Performance:")
logger.info("------------------")
logger.info(f"Accuracy  (train): {gb_tuned_metrics['Accuracy_train']:.4f}")
logger.info(f"Accuracy  (test) : {gb_tuned_metrics['Accuracy_test']:.4f}")
logger.info(f"Precision (train): {gb_tuned_metrics['Precision_train']:.4f}")
logger.info(f"Precision (test) : {gb_tuned_metrics['Precision_test']:.4f}")
logger.info(f"Recall    (train): {gb_tuned_metrics['Recall_train']:.4f}")
logger.info(f"Recall    (test) : {gb_tuned_metrics['Recall_test']:.4f}")
logger.info(f"F1 Score  (train): {gb_tuned_metrics['F1_train']:.4f}")
logger.info(f"F1 Score  (test) : {gb_tuned_metrics['F1_test']:.4f}") 

logger.info("[INFO] Most important features from XGBoost Classifier model:")
importances = best_gb_model.feature_importances_
feature_names = X_train_class.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feature_importances.to_csv(f"results/xgb_classifier_importances_{args.encoder}.csv")
logger.info(f"Saved feature importances to results/xgb_classifier_importances_{args.encoder}.csv")

logger.info("\n[INFO] Tuning XGBoost regression model...\n")
best_xgb_model, xgb_tuned_metrics = tune_xgboost(X_train, y_train, X_test, y_test, xgb_tuned_regressiom_params)
logger.info("[INFO] XGBoost tuning completed.\n")

logger.info("\nModel Performance:")
logger.info("------------------")
logger.info(f"MAE  (train): {xgb_tuned_metrics['MAE_train']:.2f}")
logger.info(f"MAE  (test) : {xgb_tuned_metrics['MAE_test']:.2f}")
logger.info(f"MSE  (train): {xgb_tuned_metrics['MSE_train']:.2e}")
logger.info(f"MSE  (test) : {xgb_tuned_metrics['MSE_test']:.2e}")
logger.info(f"R²   (train): {xgb_tuned_metrics['R2_train']:.4f}")
logger.info(f"R²   (test) : {xgb_tuned_metrics['R2_test']:.4f}")

logger.info("[INFO] Most important features from XGBoost regression model:")
importances = best_xgb_model.feature_importances_
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feature_importances.to_csv(f"results/xgb_regressor_importances_{args.encoder}.csv")
logger.info(f"Saved feature importances to results/xgb_regressor_importances_{args.encoder}.csv")

################################################
logger.info("[INFO] Preparing CatBoost datasets (raw categoricals)...")
train_cat = processor.catboost_train_data
test_cat  = processor.catboost_test_data

X_train_cat = train_cat.drop(columns=['transfer_fee'])
X_test_cat  = test_cat.drop(columns=['transfer_fee'])

y_train_cat = np.log1p(pd.to_numeric(train_cat['transfer_fee'], errors='coerce'))
y_test_cat  = np.log1p(pd.to_numeric(test_cat['transfer_fee'], errors='coerce'))

logger.info(f'X_train_cat: {X_train_cat.shape}, X_test_cat: {X_test_cat.shape}, y_train_cat: {y_train_cat.shape}, y_test_cat: {y_test_cat.shape}')
logger.info(f"[INFO] Number of categorical feature:{len(cat_var)}")
cat_features_idx = [X_train_cat.columns.get_loc(col) for col in cat_var if col in X_train_cat.columns]
logger.info("[INFO] Running baseline CatBoost model ...")
run_baseline_catboost(X_train_cat, X_test_cat, y_train_cat, y_test_cat, cat_var)
logger.info("[DONE] Baseline CatBoost results computed.\n")

logger.info("[INFO] Tuning CatBoost model ...\n")
best_cat_model, cat_metrics = tune_catboost(X_train_cat, y_train_cat, X_test_cat, y_test_cat, cat_var, catboost_tuned_params)
logger.info("[INFO] CatBoost tuning completed.\n")

logger.info("\nModel Performance:")
logger.info("------------------")
logger.info(f"MAE  (train): {cat_metrics['MAE_train']:.2f}")
logger.info(f"MAE  (test) : {cat_metrics['MAE_test']:.2f}")
logger.info(f"MSE  (train): {cat_metrics['MSE_train']:.2e}")
logger.info(f"MSE  (test) : {cat_metrics['MSE_test']:.2e}")
logger.info(f"R²   (train): {cat_metrics['R2_train']:.4f}")
logger.info(f"R²   (test) : {cat_metrics['R2_test']:.4f}")

logger.info("[INFO] Most important features from CatBoost model:")
importances = best_cat_model.feature_importances_
feature_names = X_train_cat.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feature_importances.to_csv(f"results/catboost_importances_{args.encoder}.csv")
logger.info(f"Saved feature importances to results/catboost_importances_{args.encoder}.csv")


################################################