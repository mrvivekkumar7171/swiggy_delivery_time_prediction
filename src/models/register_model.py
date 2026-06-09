from pathlib import Path
import mlflow.client
import dagshub
import logging
import mlflow
import json
import dagshub


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub & set the mlflow tracking server
dagshub.init(repo_owner='mrvivekkumar7171', repo_name='swiggy_delivery_time_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mrvivekkumar7171/swiggy_delivery_time_prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    model_info = load_model_information(run_info_path)
    
    # get the run id & model to register path
    model_uri = model_info["model_path"]
    model_name = "SwiggyDeliveryTimePredictor"
    
    
    # register the model
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name).version

    logger.info(f"The latest model version in model registry is {model_version}")

    # update the stage of the model to staging
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(name=model_name, version=model_version, alias='challenger')
    
    logger.info("Model pushed to Staging stage")
