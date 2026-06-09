# pytest tests/test_model_registry.py
from mlflow import MlflowClient
import dagshub
import pytest
import mlflow
import json

dagshub.init(repo_owner='mrvivekkumar7171', repo_name='swiggy_delivery_time_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mrvivekkumar7171/swiggy_delivery_time_prediction.mlflow")

model_name = 'SwiggyDeliveryTimePredictor'
alias = "challenger"


@pytest.mark.parametrize(argnames="model_name, alias", argvalues=[(model_name, alias)])
def test_load_model_from_registry(model_name, alias):

    # load the latest model from model registry
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    
    assert model is not None, "Failed to load model from registry"
    print(f"The {model_name} model with alias {alias} was loaded successfully")