# Importing necessary modules from FastAPI and other libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from feature_engineer import FeatureEngineer  # Custom module for feature engineering 


# Creating an instance of the FastAPI application
app = FastAPI()

# Loading the trained machine learning pipeline from a file
model = joblib.load('car_price_stacked_pipeline.pkl')

# Defining the expected input structure for a car using Pydantic for validation
class CarInput(BaseModel):
    make: str
    model: str
    fuel_type: str
    transmission: str
    year: int
    Kilometer: float

# Creating a POST endpoint at '/predict' that takes a CarInput object
@app.post("/predict")
def predict(car: CarInput):
    
    # Convert the input object to a pandas DataFrame
    df = pd.DataFrame([car.dict()])
    
    # Predict the log-transformed price using the loaded model
    log_price = model.predict(df)[0]

    # Convert the log price back to the original price scale
    price = np.expm1(log_price)

    # Return the predicted price rounded to 2 decimal places
    return {"predicted_price": round(float(price), 2)}
