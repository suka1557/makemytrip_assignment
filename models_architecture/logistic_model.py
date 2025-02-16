import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
import os, sys
from itertools import product

sys.path.append("./")

from src.balance_data import undersample_majority_class
from configs.main_config import DATA_PATH


def tune_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Trains multiple Logistic Regression models using training data and evaluates them on validation data.
    Finds the top 3 hyperparameter configurations based on F1 score on validation data.

    Parameters:
    - X_train, y_train: Training set
    - X_val, y_val: Validation set

    Returns:
    - top_3_results (pd.DataFrame): Top 3 hyperparameter configurations ranked by validation F1 score
    - best_model (LogisticRegression): Best model trained on training data
    """
    # Define Hyperparameter Grid
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        "penalty": ["l1", "l2"]  # Type of regularization
    }
    
    # Generate all possible hyperparameter combinations
    param_combinations = list(product(param_grid["C"], param_grid["penalty"]))

    # Store Results
    results = []

    # Train & Evaluate Each Model
    for C, penalty in param_combinations:
        try:
            model = LogisticRegression(C=C, penalty=penalty, solver="liblinear", random_state=42)
            model.fit(X_train, y_train)

            # Predict on Validation Set
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            # Calculate in Training Set to see if there is overfitting
            y_pred_train = model.predict(X_train)
            f1_train = f1_score(y_train, y_pred_train)
            precision_train = precision_score(y_train, y_pred_train, zero_division=0)
            recall_train = recall_score(y_train, y_pred_train, zero_division=0)

            # Store Results
            results.append({"C": C, "penalty": penalty, "val_f1": f1, "val_precision": precision, "val_recall": recall,
                            "train_f1":f1_train, "train_precision":precision_train, "train_recall":recall_train})
        except Exception as e:
            print(f"Skipping C={C}, penalty={penalty} due to error: {e}")

    # Convert Results to DataFrame & Sort by F1 Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="val_f1", ascending=False)

    # Select Top 3 Configurations
    top_3_results = results_df.head(3)

    print(top_3_results)

    return top_3_results, top_3_results["val_recall"][0]


if __name__ == '__main__':
    # Read Data and train model
    train = pd.read_feather( os.path.join(DATA_PATH, "train_processed_without_city_pair.feather") )
    val = pd.read_feather( os.path.join(DATA_PATH, "val_processed_without_city_pair.feather") )

    # Drop any rows that are null
    train.dropna(inplace=True)
    val.dropna(inplace=True)

    print(f"Records in Training = {len(train)}")

    x_train , y_train = train.drop(["ACTIVITY_TYPE"], axis=1), train["ACTIVITY_TYPE"]
    print(y_train.value_counts() / len(y_train))

    x_val, y_val = val.drop(["ACTIVITY_TYPE"], axis=1), val["ACTIVITY_TYPE"]

    #Balance train data
    x_train, y_train = undersample_majority_class(x_train, y_train)

    print(f"Records in Training after Balancing = {len(y_train)}")
    print(y_train.value_counts() / len(y_train))

    # Get top 3 logistic model hyperparameters
    top_3_config, top_recall = tune_logistic_regression(x_train, y_train, x_val, y_val)


