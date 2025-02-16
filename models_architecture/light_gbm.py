import numpy as np
import pandas as pd
from itertools import product
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score
import os, sys

sys.path.append("./")

from src.balance_data import undersample_majority_class
from configs.main_config import DATA_PATH

def tune_lightgbm(X_train, y_train, X_val, y_val):
    """
    Trains multiple LightGBM models using training data and evaluates them on validation data.
    Finds the top 3 hyperparameter configurations based on F1 score on validation data.

    Parameters:
    - X_train, y_train: Training set
    - X_val, y_val: Validation set

    Returns:
    - top_3_results (pd.DataFrame): Top 3 hyperparameter configurations ranked by validation F1 score
    - best_recall (float): Highest recall score among the top models
    """
    # Define Hyperparameter Grid
    param_grid = {
        "num_leaves": [30, 50, 100],  # Number of leaves in a tree
        "max_depth": [5, 10, 20, -1],  # Tree depth (-1 means no limit)
        "learning_rate": [0.01, 0.05, 0.1],  # Learning rate for boosting
        "n_estimators": [50, 100, 200, 500],  # Number of boosting rounds
        "min_data_in_leaf": [20, 50, 100],  # Minimum number of samples in a leaf
        "lambda_l1": [0, 0.1, 1],  # L1 regularization
        "lambda_l2": [0, 0.1, 1],  # L2 regularization
    }

    # Generate all possible hyperparameter combinations
    param_combinations = list(product(param_grid["num_leaves"], param_grid["max_depth"], 
                                      param_grid["learning_rate"], param_grid["n_estimators"],
                                      param_grid["min_data_in_leaf"], param_grid["lambda_l1"], 
                                      param_grid["lambda_l2"]))

    # Store Results
    results = []

    # Train & Evaluate Each Model
    for num_leaves, max_depth, learning_rate, n_estimators, min_data_in_leaf, lambda_l1, lambda_l2 in param_combinations:
        try:
            model = lgb.LGBMClassifier(
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                min_data_in_leaf=min_data_in_leaf,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                random_state=42,
                n_jobs=-1  # Use all available CPU cores
            )
            model.fit(X_train, y_train)

            # Predict on Validation Set
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            # Calculate on Training Set to detect overfitting
            y_pred_train = model.predict(X_train)
            f1_train = f1_score(y_train, y_pred_train, zero_division=0)
            precision_train = precision_score(y_train, y_pred_train, zero_division=0)
            recall_train = recall_score(y_train, y_pred_train, zero_division=0)

            # Store Results
            results.append({
                "num_leaves": num_leaves, "max_depth": max_depth,
                "learning_rate": learning_rate, "n_estimators": n_estimators,
                "min_data_in_leaf": min_data_in_leaf, "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "val_f1": f1, "val_precision": precision, "val_recall": recall,
                "train_f1": f1_train, "train_precision": precision_train, "train_recall": recall_train
            })
        except Exception as e:
            print(f"Skipping num_leaves={num_leaves}, max_depth={max_depth}, learning_rate={learning_rate}, "
                  f"n_estimators={n_estimators}, min_data_in_leaf={min_data_in_leaf}, lambda_l1={lambda_l1}, "
                  f"lambda_l2={lambda_l2} due to error: {e}")

    # Convert Results to DataFrame & Sort by F1 Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="val_f1", ascending=False)

    # Select Top 3 Configurations
    top_3_results = results_df.head(3)

    print("\n🏆 Top 3 Hyperparameter Configurations:\n", top_3_results)

    return top_3_results, top_3_results["val_recall"].iloc[0]

