import dagshub
import mlflow
import json

dagshub.init(repo_owner='mrvivekkumar7171', repo_name='swiggy_delivery_time_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mrvivekkumar7171/swiggy_delivery_time_prediction.mlflow")

# get model name
model_name = 'SwiggyDeliveryTimePredictor'
alias = "challenger"
promotion_alias = "champion"
demotion_alias = "archive"
client = mlflow.MlflowClient()

challenger_version = client.get_model_version_by_alias(model_name, alias).version
champion_version = client.get_model_version_by_alias(model_name, promotion_alias).version

client.set_registered_model_alias(model_name, demotion_alias, champion_version)
client.set_model_version_tag(name=model_name, version=champion_version, key="Stage", value="Archived")
client.set_registered_model_alias(model_name, promotion_alias, challenger_version)
client.set_model_version_tag(name=model_name, version=challenger_version, key="Stage", value="Production")
client.delete_registered_model_alias(model_name, alias)