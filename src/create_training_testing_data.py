import os, sys

sys.path.append("./")

from src.prepare_data import PrepareTrainTest
from configs.main_config import (
    PAST_7_DAYS_ACTIVITY_FIELD,
    TRAIN_FILENAME,
    NEXT_7_DAYS_ACTIVITY_FIELD,
    TEST_FILENAME
)

if __name__ == '__main__':
    #Initalize object
    data_preparation = PrepareTrainTest()

    #Create Training Data
    data_preparation._create_feather_dataframe(filepath=TRAIN_FILENAME, ACTIVITY_KEY=PAST_7_DAYS_ACTIVITY_FIELD)

    #Create Testing Data
    data_preparation._create_feather_dataframe(filepath=TEST_FILENAME, ACTIVITY_KEY=NEXT_7_DAYS_ACTIVITY_FIELD)
