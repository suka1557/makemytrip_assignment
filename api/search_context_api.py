import os, sys
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

sys.path.append("./")

from src.prepare_data import PrepareTrainTest
from utils.add_sin_cos_columns import add_sin_cos
from src.target_encoding import encode_target

from configs.main_config import (
    SCALING_PIPELINE,
    FINAL_MODEL,
    SIN_COS_COLUMNS,
    API_FIELD_NAMES,
    COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY,
)

#Get Data Preparation Object
data_preparation = PrepareTrainTest()
city_pairs_map = data_preparation._get_city_pairs()

# Get Model and Pipeline assets
scaling_pipeline = joblib.load(filename=SCALING_PIPELINE)
model = joblib.load(filename=FINAL_MODEL)

app = FastAPI()

class RequestPayload(BaseModel):
    features: List[float]


@app.get("/get_api_health")
async def health():
    return {"Status": "Api Running healthy"}

@app.post("/predict")
async def predict(
        FROM_CITY: int = Query(..., description="Origin city ID"),
        TO_CITY: int = Query(..., description="Destination city ID"),
        payload: RequestPayload = None
    ):

    if len(payload.features) != 23:
        raise HTTPException(status_code=400, detail="features must be a list of 23 floating point numbers")
    
    search_df = pd.DataFrame([ np.array(payload.features) ])
    search_df.columns = API_FIELD_NAMES
    search_df['FROM_TO_CITY'] = str(FROM_CITY) + "_" + str(TO_CITY)

    search_df['FROM_TO_CITY'] = search_df['FROM_TO_CITY'].apply(lambda x: city_pairs_map[x] if x in city_pairs_map else city_pairs_map['UNKNOWN'])
    search_df['FROM_TO_CITY'] = search_df['FROM_TO_CITY'].astype(int)

    #Reorder
    search_df = search_df[ ['FROM_TO_CITY'] + API_FIELD_NAMES]

    # add sin - cos columns
    for col, max_val in SIN_COS_COLUMNS.items():
        test = add_sin_cos(col, max_val, search_df)

    # Separate out FROM_TO_CITY column
    from_to_city = test['FROM_TO_CITY']
    search_df.drop(['FROM_TO_CITY'], axis = 1, inplace = True)

    # Subset df to remove some columns who have high multicollinearity
    search_df = search_df[COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY]

    #Scale data
    search_df =  pd.DataFrame( 
                scaling_pipeline.transform( search_df ),
                columns=COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY
            )
    
    #Add city pair info back
    search_df['FROM_TO_CITY'] = from_to_city

    #Reorder df
    search_df = search_df[ ['FROM_TO_CITY'] + COLUMNS_AFTER_REMOVING_MULTICOLLINEARITY ]

    #Encode target
    search_df = encode_target(search_df)

    #Make predictionds
    booking_probability = model.predict_proba(test)[:, 1].flatten()[0]
    
    return {"message": f"Booking Probability = {booking_probability}"}

