import numpy as np
import pandas as pd
from itertools import product
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def tune_xgboost(X_train, y_train, X_val, y_val):
    """
    Trains multiple XGBoost models using training data and evaluates them on validation data.
    Finds the top 3 hyperparameter configurations based on F1 score on validation data.

    Parameters:
    - X_train, y_train: Training set
    - X_val, y_val: Validation set

    Returns:
    - top_3_results (pd.DataFrame): Top 3 hyperparameter configurations ranked by validation F1 score
    - best_val_recall (float): Best recall score on validation data
    """
    # Define Hyperparameter Grid
    param_grid = {
        "n_estimators": [50, 100, 200],  # Number of boosting rounds
        "max_depth": [3, 5, 10],  # Maximum depth of trees
        "learning_rate": [0.01, 0.1, 0.2],  # Step size shrinkage
        "subsample": [0.7, 0.9, 1.0],  # Fraction of samples used for training
        "colsample_bytree": [0.7, 0.9, 1.0],  # Fraction of features used per tree
        "gamma": [0, 1, 5]  # Minimum loss reduction to make a split
    }

    # Generate all possible hyperparameter combinations
    param_combinations = list(product(param_grid["n_estimators"], param_grid["max_depth"], 
                                      param_grid["learning_rate"], param_grid["subsample"],
                                      param_grid["colsample_bytree"], param_grid["gamma"]))

    # Store Results
    results = []

    # Train & Evaluate Each Model
    for n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma in param_combinations:
        try:
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X_train, y_train)

            # Predict on Validation Set
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            # Predict on Training Set to detect overfitting
            y_pred_train = model.predict(X_train)
            f1_train = f1_score(y_train, y_pred_train, zero_division=0)
            precision_train = precision_score(y_train, y_pred_train, zero_division=0)
            recall_train = recall_score(y_train, y_pred_train, zero_division=0)

            # Store Results
            results.append({
                "n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate,
                "subsample": subsample, "colsample_bytree": colsample_bytree, "gamma": gamma,
                "val_f1": f1, "val_precision": precision, "val_recall": recall,
                "train_f1": f1_train, "train_precision": precision_train, "train_recall": recall_train
            })
        except Exception as e:
            print(f"Skipping n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, "
                  f"subsample={subsample}, colsample_bytree={colsample_bytree}, gamma={gamma} due to error: {e}")

    # Convert Results to DataFrame & Sort by F1 Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="val_f1", ascending=False)

    # Select Top 3 Configurations
    top_3_results = results_df.head(3)

    print("\n🏆 Top 3 XGBoost Hyperparameter Configurations:\n", top_3_results)

    return top_3_results, top_3_results["val_recall"].iloc[0]
