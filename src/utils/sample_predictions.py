from pathlib import Path
import pandas as pd
import requests

root_path = Path(__file__).parent.parent.parent
data_path = root_path/"data"/"raw"/"swiggy.csv"

predict_url = "http://localhost:8080/predict"

sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()
print(data)

response = requests.post(url=predict_url,json=data)

print("The status code for response is", response.status_code)

if response.status_code == 200:
    print(f"The prediction value by the API is {float(response.text):.2f} min")
else:
    print("Error:", response.status_code)