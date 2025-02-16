import joblib
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import os, sys

sys.path.append("./")

from configs.main_config import TARGET_ENCODING, DATA_PATH

def add_noise(df, noise_level=0.05):
    df['FROM_TO_CITY'] += np.random.normal(0, noise_level, df['FROM_TO_CITY'].shape)
    return df

def encode_target(df):
    encoder = joblib.load(TARGET_ENCODING)

    #Transform
    # Apply Encoding
    X_new_encoded = pd.DataFrame( encoder.transform(df), columns=df.columns)

    # Aplly noise
    X_new_encoded = add_noise(X_new_encoded)

    return X_new_encoded



if __name__ == "__main__":

    if not os.path.exists(TARGET_ENCODING):
        #Create Encoding by training on train set

        train = pd.read_feather(os.path.join(DATA_PATH, "train_processed_with_city_pair.feather"))
        # val = pd.read_feather(os.path.join(DATA_PATH, "train_processed_with_city_pair.feather"))

        # Drop Na
        train.dropna(inplace=True)
        # val.dropna(inplace=True)

        # Split Data
        X_train = train.drop(columns=["ACTIVITY_TYPE"])
        y_train = train["ACTIVITY_TYPE"]

        # Initialize Target Encoder
        encoder = TargetEncoder(cols=['FROM_TO_CITY'], handle_unknown='value', handle_missing='value')  # Unseen categories -> Mean

        # Fit encoder
        encoder.fit(X_train, y_train)

        # Save Encoder using joblib
        joblib.dump(encoder, TARGET_ENCODING)
