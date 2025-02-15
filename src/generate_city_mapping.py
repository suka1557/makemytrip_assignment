import os, sys
import pandas as pd
import pyarrow.parquet as pq
import json
import numpy as np

sys.path.append("./")

from configs.main_config import (
    INPUT_DATA_FILE,
    PAST_7_DAYS_ACTIVITY_FIELD,
    CITY_PAIRS_MAPPING_FILE,
)

def generate_from_to_city_mappings(key = PAST_7_DAYS_ACTIVITY_FIELD):
    from_to_city_pairs = {}
    
    # Open the Parquet file
    parquet_file = pq.ParquetFile(INPUT_DATA_FILE)

    print(f"Total Lines in file: {parquet_file.metadata.num_rows}")


    # Read line by line
    line_counts = 0
    for batch in parquet_file.iter_batches(batch_size=1):  # Read one row at a time
        row = batch.to_pandas().iloc[0].to_dict()  # Convert to dictionary
        sample = row['sample']

        activities = sample[key]

        for activity in activities:
            
            from_city = activity['from_id']
            to_city = activity['to_id']
            from_to_city = str(from_city) + "_" + str(to_city)
            if from_to_city not in from_to_city_pairs:
                from_to_city_pairs[from_to_city] = len(from_to_city_pairs)
            

        line_counts += 1

        if line_counts % 1000 == 0:
            print(f"Processed {line_counts} lines in parquet file")

    #Add 1 field to handle unknown city pairs while inferencting
    from_to_city_pairs["UNKNOWN"] = len(from_to_city_pairs)

    print(f"Total City Pairs in {PAST_7_DAYS_ACTIVITY_FIELD} activity = {len(from_to_city_pairs)}")

    #Save the dictinory to json
    with open(CITY_PAIRS_MAPPING_FILE, "w") as file:
        json.dump(from_to_city_pairs, file)

if __name__ == '__main__':
    generate_from_to_city_mappings()
            
