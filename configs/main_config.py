import os
import sys

PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join( PROJECT_ROOT, 'data')

MODELS_FOLDER = os.path.join(PROJECT_ROOT, 'saved_models')
EMBEDDINGS_FOLDER = os.path.join(PROJECT_ROOT, 'saved_embeddings')

# Make folders if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

INPUT_DATA_FILE = os.path.join(DATA_PATH, 'assignment.parquet')
CITY_PAIRS_MAPPING_FILE = os.path.join(DATA_PATH, 'city_pairs_map.json')


FIELD_NAMES = [
    'LAT_FROM_CITY', 'LON_FROM_CITY', 'AREA_FROM_CITY', 'LAT_TO_CITY','LON_TO_CITY','AREA_TO_CITY',
    'DAYS_AFTER_01JAN2023',
    'TRAVEL_DAY_OF_WEEK', 'TRAVEL_DAY_OF_MONTH','TRAVEL_DAY_OF_YEAR',
    'SEARCH_TIMESTAMP_AFTER_01JAN2023',
    'OTHER1','OTHER2','OTHER3','OTHER4',
    'ACTIVITY_TYPE',
    'PRICE1','PRICE2','PRICE3','PRICE4','PRICE5','PRICE6','PRICE7','PRICE8'
]

TARGET_COLUMN_NAME = 'ACTIVITY_TYPE'
NUMERICAL_COLUMNS = [col for col in FIELD_NAMES if col != TARGET_COLUMN_NAME]
COUNT_NUMERICAL_COLUMNS = len(NUMERICAL_COLUMNS)

PAST_7_DAYS_ACTIVITY_FIELD = 'context'
NEXT_7_DAYS_ACTIVITY_FIELD = 'affinity'

TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
BATCH_SIZE = 1028

BATCH_SIZE = 1028
NO_OF_CITY_PAIRS = 4692
CITY_PAIR_EMBEDDING_DIMENSION = 300

MODEL_NAME = 'model_' + str( len( os.listdir(MODELS_FOLDER) ) + 1 )
EMBEDDING_NAME = 'embeddings_' + str(CITY_PAIR_EMBEDDING_DIMENSION) + '_' + str( len(os.listdir(EMBEDDINGS_FOLDER)) + 1)
