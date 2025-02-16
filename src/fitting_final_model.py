import os, sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

sys.path.append("./")

from src.target_encoding import encode_target
from configs.main_config import DATA_PATH, FINAL_MODEL
from src.balance_data import undersample_majority_class

if __name__ == "__main__":

    # Read Data and train model
    train = pd.read_feather( os.path.join(DATA_PATH, "train_processed_with_city_pair.feather") )
    val = pd.read_feather( os.path.join(DATA_PATH, "val_processed_with_city_pair.feather") )
    training_data = pd.concat([train, val], ignore_index=True)
    training_data = training_data.reset_index(drop=True)

    # Drop any rows that are null
    training_data.dropna(inplace=True)
    print(f"Records in Training = {len(training_data)}")

    x_train , y_train = training_data.drop(["ACTIVITY_TYPE"], axis=1), training_data["ACTIVITY_TYPE"]
    print(y_train.value_counts() / len(y_train))

    #Apply target encoding
    x_train = encode_target(x_train)

    #Balance train data
    x_train, y_train = undersample_majority_class(x_train, y_train)

    #Declare and Fit model
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=50,
        criterion="gini",
        random_state=42
    )

    model.fit(x_train, y_train)

    #save final model
    joblib.dump(model, FINAL_MODEL)


