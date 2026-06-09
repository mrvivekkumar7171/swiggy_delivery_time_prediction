from mlflow import MlflowClient
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
    run_info = load_model_information(run_info_path)
    
    # get the run id
    run_id = run_info["run_id"]
    artifact_path = run_info["artifact_path"]
    model_name = run_info["model_name"]
    
    # model to register path
    model_registry_path = f"runs:/{run_id}/{artifact_path}"
    
    
    # register the model
    print(f"Registering model from exact URI: {model_registry_path}")
    model_version = mlflow.register_model(model_uri=model_registry_path, name=model_name)
    
    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")
    
    # update the stage of the model to staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )
    
    logger.info("Model pushed to Staging stage")
