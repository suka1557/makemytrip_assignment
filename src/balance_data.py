import os, sys
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler


sys.path.append("./")

def undersample_majority_class(x_df, y_target, sampling_strategy="auto", random_state=100):
    """
    Performs random undersampling to balance the dataset.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    x_df_balanced, y_target_balanced = rus.fit_resample(x_df, y_target)
    
    return pd.DataFrame(x_df_balanced, columns=x_df.columns), pd.Series(y_target_balanced)
