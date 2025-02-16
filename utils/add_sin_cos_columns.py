import numpy as np

def add_sin_cos(col, max_value, df):

    #Create sine
    df[col+"_sin"] = np.sin(2 * np.pi * df[col] / max_value)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_value)

    # Drop original column
    df.drop([col], axis=1, inplace=True)

    return df