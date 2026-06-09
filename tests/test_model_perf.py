# pytest tests/test_model_perf.py
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
import dagshub
import joblib
import pytest
import mlflow
import json

dagshub.init(repo_owner='mrvivekkumar7171', repo_name='swiggy_delivery_time_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mrvivekkumar7171/swiggy_delivery_time_prediction.mlflow")


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

model_name = 'SwiggyDeliveryTimePredictor'
alias = "challenger"
model_uri = f"models:/{model_name}@{alias}"
model = mlflow.sklearn.load_model(model_uri=model_uri)

# set the root path & load the preprocessor
root_path = Path(__file__).parent.parent
preprocessor_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ("regressor", model)
])
test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe, test_data_path, threshold_error", argvalues=[(model_pipe, test_data_path, 5)])
def test_model_performance(model_pipe, test_data_path, threshold_error):
    # load test data & drop the missing values
    df = pd.read_csv(test_data_path)
    df.dropna(inplace=True)

    # make X and y
    X = df.drop(columns=["time_taken"])
    y = df['time_taken']

    # get the predictions & calculate the mean error
    y_pred = model_pipe.predict(X)
    mean_error = mean_absolute_error(y, y_pred)

    # check for performance
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes"
    print("The avg error is", mean_error)

    print(f"The {model_name} model passed the performance test")