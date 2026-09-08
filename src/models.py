import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor

# ===============================================================
# SKLEARN WRAPPER
# ===============================================================

class SklearnModel:
    def __init__(self, name, estimator):
        self.name = name
        self.estimator = estimator

    def fit(self, X, y):
        print(f"      > Fitting model '{self.name}' ...")
        self.estimator.fit(X, y)

    def predict(self, X):
        print(f"      > Predicting with model '{self.name}' ...")
        return self.estimator.predict(X)


# ===============================================================
# EVALUATION
# ===============================================================

def evaluate_regression(model, X_train, y_train, X_test, y_test):

    print(f"\n[MODEL] Running evaluation for: {model.name}")
    model.fit(X_train, y_train)

    pred_train_log = model.predict(X_train)
    pred_test_log  = model.predict(X_test)

    print("      > Computing metrics ...")
    metrics = {
        "MAE_train": mean_absolute_error(y_train, pred_train_log),
        "MAE_test": mean_absolute_error(y_test, pred_test_log),
        "MSE_train": mean_squared_error(y_train, pred_train_log),
        "MSE_test": mean_squared_error(y_test, pred_test_log),
        "R2_train": r2_score(y_train, pred_train_log),
        "R2_test":  r2_score(y_test, pred_test_log),
    }

    print("      > Metrics computed successfully.")
    return metrics

def evaluate_classification(model, X_train, y_train, X_test, y_test):

    print(f"\n[MODEL] Running evaluation for: {model.name}")
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    print("      > Computing metrics ...")
    metrics = {
        "Accuracy_train": accuracy_score(y_train, pred_train),
        "Accuracy_test": accuracy_score(y_test, pred_test),
        "Precision_train": precision_score(y_train, pred_train, average='weighted', zero_division=0),
        "Precision_test": precision_score(y_test, pred_test, average='weighted', zero_division=0),
        "Recall_train": recall_score(y_train, pred_train, average='weighted', zero_division=0),
        "Recall_test": recall_score(y_test, pred_test, average='weighted', zero_division=0),
        "F1_train": f1_score(y_train, pred_train, average='weighted', zero_division=0),
        "F1_test": f1_score(y_test, pred_test, average='weighted', zero_division=0),
    }

    print("      > Metrics computed successfully.")
    return metrics
# ===============================================================
# MODEL DEFINITIONS
# ===============================================================

def get_all_models():
    print("\n[INFO] Building model list ...")

    models = [
        SklearnModel(
            name="LinearRegression",
            estimator=LinearRegression(),
        ),

        SklearnModel(
            name="RandomForest",
            estimator=RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        ),

        SklearnModel(
            name="GradientBoosting",
            estimator=GradientBoostingRegressor()
        ),

        SklearnModel(
            name="XGBoost",
            estimator=xgb.XGBRegressor(
                n_jobs=-1,
                random_state=42,
            )
        ),
    ]

    print("[INFO] Models ready.\n")
    return models

def get_all_classification_models():
    print("\n[INFO] Building classification model list ...")

    models = [
        SklearnModel("LogisticRegression", LogisticRegression(max_iter=1500)),

        SklearnModel(
            "RandomForestClassifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        ),

        SklearnModel(
            "XGBoostClassifier",
            xgb.XGBClassifier(
                eval_metric='logloss',
                random_state=42
            )
        ),

        SklearnModel(
            "GradientBoostingClassifier",
            GradientBoostingClassifier()
        ),
    ]

    print("[INFO] Classification models ready.\n")
    return models


# ===============================================================
# RUNNER
# ===============================================================

class ModelRunner:
    def __init__(self, models, evaluator=evaluate_regression):
        self.models = models
        self.evaluator = evaluator

    def run(self, X_train, y_train, X_test, y_test):
        print("\n[RUNNER] Starting model evaluations ...\n")

        results = []
        for model in self.models:
            print(f"[RUNNER] Evaluating model: {model.name}")
            metrics = self.evaluator(model, X_train, y_train, X_test, y_test)
            metrics["model"] = model.name
            results.append(metrics)
            print(f"[RUNNER] Finished model: {model.name}\n")

        print("[RUNNER] All models evaluated.\n")
        return pd.DataFrame(results).set_index("model")

    def print_results(self, df):
        print("\n================ FINAL MODEL RESULTS ================\n")
        for model_name, row in df.iterrows():
            print(f"--- {model_name} ---")
            for metric, value in row.items():
                print(f"{metric}: {value:.4f}")
            print()
        print("=====================================================\n")


# ===============================================================
# XGBOOST TUNING
# ===============================================================

def tune_xgboost(X_train, y_train, X_test, y_test, xgb_params):

    print("\n======================= XGBOOST HYPERPARAMETER TUNING (done before now just evaluate on the best params)=======================\n")


    param_dist_xgb = {
        "n_estimators": [100, 200, 300, 500, 700, 1000],
        "learning_rate": np.linspace(0.01, 0.3, 20),
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": [1, 2, 3, 4, 5],
        "subsample": np.linspace(0.5, 1.0, 6),
        "colsample_bytree": np.linspace(0.5, 1.0, 6),
        "gamma": [0, 0.1, 0.2, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1, 10],
        "reg_lambda": [0.1, 1, 5, 10, 50]
    }
    
    # param_dist_xgb documents the search space explored during tuning; the
    # winning combination is passed in as xgb_params and evaluated directly
    # below, so RandomizedSearchCV does not need to run again here.
    best_model = xgb.XGBRegressor(
        **xgb_params,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror',
        enable_categorical=False
    )

    random_metrics = evaluate_regression(
        SklearnModel("XGBoost_RandomSearch", best_model),
        X_train, y_train, X_test, y_test
    )

    return best_model, random_metrics

def tune_xgboost_classifier(X_train_class, y_train_class, X_test_class, y_test_class, xgb_class_params):
    print("\n======================= XGBOOST CLASSIFICATION HYPERPARAMETER TUNING (done before now just evaluating) =======================\n")
    param_dist_xgb = {
        "n_estimators": [300, 500, 600, 800],
        "learning_rate": np.linspace(0.01, 0.3, 15),
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": [3, 4, 5, 6],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1, 10],
        "reg_lambda": [0.1, 1, 5, 10, 50]  
    }
    # param_dist_xgb documents the search space explored during tuning; the
    # winning combination is passed in as xgb_class_params and evaluated
    # directly below, so RandomizedSearchCV does not need to run again here.
    best_model = xgb.XGBClassifier(
        **xgb_class_params,
        random_state=42,
        n_jobs=-1,
        objective='multi:softprob'
    )

    random_metrics = evaluate_classification(
        SklearnModel("GradientBoostingClassifier", best_model),
        X_train_class, y_train_class, X_test_class, y_test_class
    )

    return best_model, random_metrics

# ===============================================================
# CATBOOST 
# ===============================================================

def run_baseline_catboost(X_train_cat, X_test_cat, y_train_cat, y_test_cat, cat_vars):
    baseline_cat = CatBoostRegressor(
        random_seed=42,
        silent=True,
        cat_features=cat_vars
    )

    baseline_cat.fit(X_train_cat, y_train_cat)

    pred_train_log = baseline_cat.predict(X_train_cat)
    pred_test_log  = baseline_cat.predict(X_test_cat)

    baseline_metrics = {
        "MAE_train": mean_absolute_error(y_train_cat, pred_train_log),
        "MAE_test": mean_absolute_error(y_test_cat, pred_test_log),
        "MSE_train": mean_squared_error(y_train_cat, pred_train_log),
        "MSE_test": mean_squared_error(y_test_cat, pred_test_log),
        "R2_train": r2_score(y_train_cat, pred_train_log),
        "R2_test":  r2_score(y_test_cat, pred_test_log)
    }
    print("\n=== BASELINE CATBOOST RESULTS ===")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v:.4f}")
    print("=================================\n")

    
def tune_catboost(X_train, y_train, X_test, y_test, cat_features, catboost_params):

    print("\n======================= CATBOOST HYPERPARAMETER TUNING (done before, now just evaluate on the best estimators) =======================\n")

    # param_dist_catboost documents the search space explored during tuning;
    # the winning combination is passed in as catboost_params and evaluated
    # directly below, so RandomizedSearchCV does not need to run again here.
    param_dist_catboost = {
        "iterations": [1000, 1200, 1500],
        "learning_rate": np.linspace(0.01, 0.2, 20),
        "depth": [4, 5, 6, 7, 8, 9, 10],
        "subsample": np.linspace(0.5, 1.0, 6),
        "l2_leaf_reg": [11, 15, 20, 25, 30]
    }

    best_model = CatBoostRegressor(
        **catboost_params,
        loss_function="RMSE",
        random_seed=42,
        verbose=50,
        cat_features=cat_features
    )

    random_metrics = evaluate_regression(
        SklearnModel("CatBoost_RandomSearch", best_model),
        X_train, y_train, X_test, y_test
    )

    return best_model, random_metrics

