import pandas as pd
import os, sys

sys.path.append("./")

from models_architecture.logistic_model import tune_logistic_regression
from models_architecture.decision_tree import tune_decision_tree
from models_architecture.random_forest import tune_random_forest
from models_architecture.light_gbm import tune_lightgbm
from models_architecture.xgboost_model import tune_xgboost
from src.balance_data import undersample_majority_class
from configs.main_config import DATA_PATH


TUNING_FUNCTION = {
    'logistic_regression': tune_logistic_regression,
    'decision_tree': tune_decision_tree,
    'random_forest': tune_random_forest,
    'light_gbm': tune_lightgbm,
    'xgboost': tune_xgboost,
}

def tune_all_models(X_train, y_train, X_val, y_val, tuning_functions=TUNING_FUNCTION):
    """
    Runs hyperparameter tuning for multiple models and selects the best one based on validation F1 score.

    Parameters:
    - X_train, y_train: Training set
    - X_val, y_val: Validation set

    Returns:
    - best_model_name (str): Name of the best model
    - best_model_params (dict): Best hyperparameter configuration
    - all_results_df (pd.DataFrame): DataFrame with all model results
    """
    
    # Store results from each model
    all_results = []

    # Run tuning for each model
    for model_name, tuning_function in tuning_functions.items():
        print(f"🔄 Tuning {model_name}...")
        top_configs, _ = tuning_function(X_train, y_train, X_val, y_val)  # Get top 3 configs
        
        # Add model name to the results
        top_configs["model"] = model_name
        all_results.append(top_configs)

    # Combine results into a single DataFrame
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Find the best model based on validation F1 score
    best_model_row = all_results_df.loc[all_results_df["val_recall"].idxmax()]

    # Extract best model details
    best_model_name = best_model_row["model"]
    best_model_params = best_model_row.drop(["model", "val_f1", "val_precision", "val_recall", 
                                             "train_f1", "train_precision", "train_recall"]).to_dict()

    print("\n🏆 Best Model Found:")
    print(f"🔹 Model: {best_model_name}")
    print(f"🔹 Best Hyperparameters: {best_model_params}")

    return best_model_name, best_model_params


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
    best_model_name, best_model_params = tune_random_forest(x_train, y_train, x_val, y_val)
