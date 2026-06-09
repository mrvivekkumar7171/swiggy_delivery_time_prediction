from src.utils.data_clean_utils_prod import perform_data_cleaning
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import dagshub
import uvicorn
import joblib
import mlflow
import json

from sklearn import set_config
set_config(transform_output='pandas')

# initialize dagshub
dagshub.init(repo_owner='mrvivekkumar7171', repo_name='swiggy_delivery_time_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mrvivekkumar7171/swiggy_delivery_time_prediction.mlflow")


class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer


num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]
nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]
ordinal_cat_cols = ["traffic", "distance_type"]
model_name = 'SwiggyDeliveryTimePredictor'
alias = "challenger"
model_uri = f"models:/{model_name}@{alias}"
model = mlflow.sklearn.load_model(model_uri=model_uri)

preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ("regressor", model)
])

app = FastAPI()

@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        },index=[0]
    )
    cleaned_data = perform_data_cleaning(pred_data)
    return model_pipe.predict(cleaned_data)[0]
   
   
if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8080)