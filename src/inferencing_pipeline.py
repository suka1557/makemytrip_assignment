import os, sys
import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq

sys.path.append("./")

from configs.main_config import (
    SCALING_PIPELINE,
    FINAL_MODEL,
    INPUT_DATA_FILE,
    NEXT_7_DAYS_ACTIVITY_FIELD,
    OPTIMAL_THRESHOLD,
    BATCH_SIZE,
    SIN_COS_COLUMNS,
    COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY,
)
from src.prepare_data import PrepareTrainTest
from utils.add_sin_cos_columns import add_sin_cos
from src.target_encoding import encode_target

# Load the pipeline
scaling_pipeline = joblib.load(filename=SCALING_PIPELINE)
model = joblib.load(filename=FINAL_MODEL)

def create_dataframe_from_parquet(
        file_path: str,
        data_preparation_object: PrepareTrainTest,
        batch_size: int,
        ):

    parquet_file = pq.ParquetFile(file_path)
    city_pairs_map = data_preparation_object._get_city_pairs()
    
    row_count = 0
    df_list = []  # Temporary list to store processed rows

    # Iterate over batches of 1000 lines from the parquet file
    for batch in parquet_file.iter_batches(batch_size=batch_size):  
        
        for row in batch.to_pandas().to_dict(orient="records"):  # Convert batch to list of dicts
            sample = row['sample']
            activity_df = data_preparation_object._process_activity(activities_array=sample[NEXT_7_DAYS_ACTIVITY_FIELD], city_map=city_pairs_map)
            df_list.append(activity_df)  # Collect processed rows
            row_count += 1

        print(f"Processed {row_count} Rows")
        
    # Combine all expanded rows into a single dataframe
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df  = combined_df.reset_index(drop=True)

    if "ACTIVITY_TYPE" in combined_df.columns:
        combined_df.drop(["ACTIVITY_TYPE"], axis=1, inplace=True)

    return combined_df



def predict_booking(filename=INPUT_DATA_FILE, scaler=scaling_pipeline, model=model, threshold=OPTIMAL_THRESHOLD):

    data_preparation = PrepareTrainTest()
    
    # Get data frame
    test = create_dataframe_from_parquet(filename, data_preparation, BATCH_SIZE)

    # add sin - cos columns
    for col, max_val in SIN_COS_COLUMNS.items():
        test = add_sin_cos(col, max_val, test)

    # Separate out FROM_TO_CITY column
    from_to_city = test['FROM_TO_CITY']
    test.drop(['FROM_TO_CITY'], axis = 1, inplace = True)

    # Subset df to remove some columns who have high multicollinearity
    test = test[COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY]

    #Scale data
    test =  pd.DataFrame( 
                scaler.transform( test ),
                columns=COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY
            )
    
    #Add city pair info back
    test['FROM_TO_CITY'] = from_to_city

    #Reorder df
    test = test[ ['FROM_TO_CITY'] + COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY ]

    #Encode target
    test = encode_target(test)

    #Drop na
    test.dropna(inplace=True)

    #Make predictionds
    y = model.predict_proba(test)[:, 1].flatten() > threshold

    return y

if __name__ == "__main__":
    predictions = predict_booking(INPUT_DATA_FILE, scaling_pipeline, model, OPTIMAL_THRESHOLD)
    print(predictions)



