import sys, os
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import json
import torch
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append("./")

from configs.main_config import (
    FIELD_NAMES,
    DATA_PATH,
    INPUT_DATA_FILE,
    PAST_7_DAYS_ACTIVITY_FIELD,
    CITY_PAIRS_MAPPING_FILE,
    BATCH_SIZE,
    TARGET_COLUMN_NAME,
    NUMERICAL_COLUMNS,
)



class PrepareTrainTest(IterableDataset):

    def __init__(
            self, 
            input_file=INPUT_DATA_FILE, 
            column_names = FIELD_NAMES,
            data_folder = DATA_PATH,
            activity_key=PAST_7_DAYS_ACTIVITY_FIELD,
            city_pairs_file=CITY_PAIRS_MAPPING_FILE,
            batch_size = BATCH_SIZE,
            target = TARGET_COLUMN_NAME,
            numerical_columns = NUMERICAL_COLUMNS,
            ):
        
        self.input_data_file = input_file
        self.column_names = column_names
        self.data_folder = data_folder
        self.activity_key = activity_key
        self.city_pairs_file = city_pairs_file
        self.batch_size = batch_size
        self.target = target
        self.numerical_columns = numerical_columns

    def _get_city_pairs(self) -> dict:
        # Open and read the JSON file
        with open(self.city_pairs_file, 'r') as file:
            from_to_city_map = json.load(file)

        return from_to_city_map

    def _fill_missing(self, train:pd.DataFrame, val:pd.DataFrame):
        # Fill missing values
        num_imputer = train[self.numerical_columns].median().to_dict()

        train[self.numerical_columns] = train[self.numerical_columns].fillna(num_imputer)
        val[self.numerical_columns] = val[self.numerical_columns].fillna(num_imputer)

        return train, val

    def _scale(self, train:pd.DataFrame, val:pd.DataFrame):
        # Scale
        
        scaler = StandardScaler()

        train[self.numerical_columns] = scaler.fit_transform(train[self.numerical_columns])
        val[self.numerical_columns] = scaler.transform(val[self.numerical_columns])

        return train, val

        
    def _process_activity(self, activities_array: list, city_map: dict) -> pd.DataFrame:
        features_matrix = []
        from_city = []
        to_city = []

        for search_item in activities_array:
            features_array = search_item['features']
            features_matrix.append( features_array)
            
            from_city.append(str( search_item['from_id']))
            to_city.append(str( search_item['to_id']))

        df = pd.DataFrame(np.array(features_matrix))
        df.columns = self.column_names
        df['FROM_TO_CITY'] = [from_ci + "_" + to_ci for from_ci, to_ci in zip(from_city, to_city)]
        df['FROM_TO_CITY'] = df['FROM_TO_CITY'].apply(lambda x: city_map[x] if x in city_map else city_map['UNKNOWN'])
        df['FROM_TO_CITY'] = df['FROM_TO_CITY'].astype(int)

        #Reorder columns
        df = df[['FROM_TO_CITY'] +  self.numerical_columns + [self.target] ]

        # COnvert target column to Binary
        df[self.target] = np.where(df[self.target] == 3, 1, 0)

        return df
    
    def _get_X_Y(self, df:pd.DataFrame):
        X = df.drop(labels=[self.target], axis=1)
        Y = df[self.target]

        return X,Y
    
    def __iter__(self):
      parquet_file = pq.ParquetFile(self.input_data_file)
      city_pairs_map = self._get_city_pairs()

      # Iterate over batches of 1000 lines from the parquet file
      for batch in parquet_file.iter_batches(batch_size=self.batch_size):  
          df_list = []  # Temporary list to store processed rows

          for row in batch.to_pandas().to_dict(orient="records"):  # Convert batch to list of dicts
              sample = row['sample']
              activity_df = self._process_activity(activities_array=sample[self.activity_key], city_map=city_pairs_map)
              df_list.append(activity_df)  # Collect processed rows
          
          # Combine all expanded rows into a single dataframe
          batch_df = pd.concat(df_list, ignore_index=True)

          # Convert to PyTorch tensors
          X, Y = self._get_X_Y(batch_df)

          # Do train test split
          X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True, stratify=Y)

          # Fill Missing Values
          X_train, X_test = self._fill_missing(X_train, X_test)

          # Scale
          X_train, X_test = self._scale(X_train, X_test)

          # Convert dataframe to tensors
          X_train = torch.tensor(X_train.iloc[:, ].values, dtype=torch.float32)  # Features
          X_test = torch.tensor(X_test.iloc[:, ].values, dtype=torch.float32)  # Features

          y_train = torch.tensor(y_train.values, dtype=torch.long)  # Target
          y_test = torch.tensor(y_test.values, dtype=torch.long)  # Target

          yield X_train,  y_train, X_test, y_test  # ✅ Yield full batch

    def _create_feather_dataframe(self, filepath, ACTIVITY_KEY=PAST_7_DAYS_ACTIVITY_FIELD):
        parquet_file = pq.ParquetFile(self.input_data_file)
        city_pairs_map = self._get_city_pairs()
        row_count = 0
        df_list = []  # Temporary list to store processed rows

        # Iterate over batches of 1000 lines from the parquet file
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):  
            
            for row in batch.to_pandas().to_dict(orient="records"):  # Convert batch to list of dicts
                sample = row['sample']
                activity_df = self._process_activity(activities_array=sample[ACTIVITY_KEY], city_map=city_pairs_map)
                df_list.append(activity_df)  # Collect processed rows
                row_count += 1

            print(f"Processed {row_count} Rows")
            
        # Combine all expanded rows into a single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df  = combined_df.reset_index(drop=True)

        #Save to feather dataframe
        combined_df.to_feather(filepath)
        print(f"Saved {len(combined_df)} Records")





