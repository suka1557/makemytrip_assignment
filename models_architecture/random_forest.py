import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import os, sys

sys.path.append("./")

from src.balance_data import undersample_majority_class
from configs.main_config import DATA_PATH
from src.target_encoding import encode_target

def tune_random_forest(X_train, y_train, X_val, y_val):
    """
    Trains multiple Random Forest models using training data and evaluates them on validation data.
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
        "n_estimators": [50, 100, 200, 500],  # Number of trees
        "max_depth": [5, 10, 20, None],  # Tree depth
        "min_samples_split": [20, 50, 100, 200],  # Minimum samples required to split
        "min_samples_leaf": [20, 50, 100, 200],  # Minimum samples in a leaf
        "criterion": ["gini", "entropy"]  # Splitting criteria
    }

    # Generate all possible hyperparameter combinations
    param_combinations = list(product(param_grid["n_estimators"], param_grid["max_depth"], 
                                      param_grid["min_samples_split"], param_grid["min_samples_leaf"], 
                                      param_grid["criterion"]))

    # Store Results
    results = []

    # Train & Evaluate Each Model
    for n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion in param_combinations:
        try:
            print(f"Fitting Random Forest Models...")

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
                n_jobs=-1  # Use all available CPU cores for faster training
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
                "n_estimators": n_estimators, "max_depth": max_depth,
                "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
                "criterion": criterion,
                "val_f1": f1, "val_precision": precision, "val_recall": recall,
                "train_f1": f1_train, "train_precision": precision_train, "train_recall": recall_train
            })

            print(results[-1])
            
        except Exception as e:
            print(f"Skipping n_estimators={n_estimators}, max_depth={max_depth}, "
                  f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                  f"criterion={criterion} due to error: {e}")

    # Convert Results to DataFrame & Sort by F1 Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="val_f1", ascending=False)

    # Select Top 3 Configurations
    top_3_results = results_df.head(3)

    print("\n🏆 Top 3 Hyperparameter Configurations:\n", top_3_results)

    return top_3_results, top_3_results["val_recall"].iloc[0]

if __name__ == '__main__':
    # Read Data and train model
    train = pd.read_feather( os.path.join(DATA_PATH, "train_processed_with_city_pair.feather") )
    val = pd.read_feather( os.path.join(DATA_PATH, "val_processed_with_city_pair.feather") )

    # Drop any rows that are null
    train.dropna(inplace=True)
    val.dropna(inplace=True)

    print(f"Records in Training = {len(train)}")

    x_train , y_train = train.drop(["ACTIVITY_TYPE"], axis=1), train["ACTIVITY_TYPE"]
    print(y_train.value_counts() / len(y_train))

    x_val, y_val = val.drop(["ACTIVITY_TYPE"], axis=1), val["ACTIVITY_TYPE"]

    #Apply target encoding
    x_train = encode_target(x_train)
    x_val = encode_target(x_val)

    #Balance train data
    x_train, y_train = undersample_majority_class(x_train, y_train)

    print(f"Records in Training after Balancing = {len(y_train)}")
    print(y_train.value_counts() / len(y_train))

    # Get top 3 random fporest model hyperparameters
    top_3_config, top_recall = tune_random_forest(x_train, y_train, x_val, y_val)


