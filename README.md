### **Swiggy Delivery Time Prediction**

The project aims to build an ML model that will predicts **food delivery time** taken by the `rider` to deliver food from origin `restaurant` to destination `customer`.

It is a `Regression` problem where we predicts the delivery time in minutes from the restaurant to the customer. We will be using only RMSE and MAE since, we want prediction in minute instead of minute square and minutes square is not interpretable.
**This project takes three types of input features:**
- Rider Information: `rider` and `vehicle` of the rider.
- Environmental Factors: `weather`, `traffic`, `City` and `Holiday`.
- Location Information: `location` of `restaurant` and `delivery`.

### **How this model helps in improving business ?**
- **Enhances Customer Satisfaction & Trust (CSAT):** accurate estimated delivery time predictions (ETAs) allow customers to plan better, reducing anxiety and frustration. This transparency builds trust, improves brand image, and leads to higher customer satisfaction scores.
- **Increases Customer Retention & Lifetime Value (CLV):** Reliable delivery experiences encourage repeat business, directly increasing the retention rate and long-term revenue without additional acquisition costs.
- **Optimizes Resource Allocation & Scheduling:** Predictive insights allow dispatch teams to plan shifts and routes more effectively, such as assigning drivers to specific zones during predicted peak traffic or weather disruptions, reducing idle time.
- **Improves Operational Efficiency & Driver Utilization:** Enables better routing and delivery clustering (managing multiple orders), which maximizes the number of deliveries per hour and optimizes each driver’s workload.
- **Reduces Operational & Support Costs:** Accurate ETAs minimize the volume of "Where is my order?" support calls, significantly lowering customer service expenses. It also reduces fuel and labor costs through optimized routing.
- **Minimizes Order Cancellations:** With transparent and realistic wait times, customers are less likely to cancel orders due to uncertainty, directly improving the delivery success rate.
- **Boosts Average Order Value (AOV):** consistently good delivery experiences build confidence, encouraging customers to place larger or more frequent orders, especially during promotions.
- **Enables Dynamic Decision Making:** Allows for real-time adjustments, such as reassigning drivers or rescheduling deliveries during unforeseen delays, ensuring high-priority orders are still met.
- **Facilitates Strategic Planning & Expansion:** Data on delivery bottlenecks helps identify where to open new micro-fulfillment centers or hubs and aids in planning for geographic expansion.
- **Empowers Riders & Reduces Risk:** Drivers can plan their pickups and deliveries better, reducing the need for risky driving to meet unrealistic deadlines. This leads to better ratings, reduced stress, and the potential for higher earnings through efficiency.
- **Supports Restaurant Operations:** Helps restaurants balance in-house vs. delivery orders by prioritizing based on accurate pickup times. It also allows for better staff scaling during predicted demand spikes.
- **Revenue Management (Surge Pricing & Promotions):** Helps identify peak demand periods to implement surge pricing effectively or offer discounts during off-peak hours to level out demand and ensure continuous revenue generation.
- **Reduces Compensation Costs:** Proactive management of delays helps avoid costs associated with refunds, compensations, or discounts given to appease unhappy customers.

### Data Cleaning, Preprocessing and EDA
We performed detailed data cleaning, preprocessing and EDA to understand the data better and to prepare it for modeling.
- we perfrom `chi_2_test`, `anova_test` and `test_for_normality` like jarque bera test for varifying out finding before conclusion.

### Experimentation

**DagsHub**
DagsHub has been used for version control, data versioning, experiment tracking and model registry. It is a platform that provides a collaborative environment for data science and machine learning projects. It allows teams to manage their code, data, and experiments in one place, making it easier to track changes, reproduce results, and collaborate effectively.

**MLFLOW**
MLflow has been used for experiment tracking, model versioning in model registry and model deployment.

#### Baseline Model with Linear Regression
- mae has decreased from 4.70 to 4.69.on train and test dataset respectively.
- r2 score remain constant i.e. 0.60, on both train and test dataset.

#### Random Forest with Dropping Missing Values vs Imputation
**Random Forest with Dropping Missing Values**
- mae has increased from 1.15 to 3.09 on train and test dataset respectively.
- r2 score has decreased from 0.98 to 0.83 on train and test dataset respectively.

**Random Forest with Imputation**
- mae has increased from 1.22 to 3.29 on train and test dataset respectively.
- r2 score has decreased from 0.97 to 0.80 on train and test dataset respectively.
![alt text](/reports/figures/image-2.png)

Since, the r2 score after imputation is less than the r2 score after dropping the missing values and error also increases after imputation, we can conclude that dropping the missing values is a better approach for this dataset.

#### Random Forest with Imputation and Missing Value Indicator
- mae has increased from 1.21 to 3.29 on train and test dataset respectively.
- r2 score has decreased from 0.97 to 0.80 on train and test dataset respectively.
![alt text](/reports/figures/image-3.png)
Again trained the model with missing imputation and missing indicator but still the performance is not good than dropping the missing values. So, I will drop the missing values and train the model.

#### Model selection with Hyperparameter tuning
Top best models are as follows:
1. **LightGBM**
Average MAE of 3.17 achived.
- mae has increased from 2.78 to 3.03 on train and test dataset respectively.
- r2 score has decreased from 0.86 to 0.84 on train and test dataset respectively.

2. **Random Forest**
Average MAE of 3.21 achived.
![alt text](/reports/figures/image-4.png)

#### Detailed Hyperparameter Tunning of Top 2 models
1. **LightGBM**
- Average MAE of 3.17 achived.
![alt text](/reports/figures/image-6.png)
```python
{'n_estimators': 178,
 'max_depth': 30,
 'learning_rate': 0.2770325079361985,
 'subsample': 0.9234451532546651,
 'min_child_weight': 14,
 'min_split_gain': 0.011307090100112493,
 'reg_lambda': 98.1093294393446}
```

2. **Random Forest**
- Average MAE of 3.08 achived.
![alt text](/reports/figures/image-8.png)
```python
{'n_estimators': 344,
 'criterion': 'squared_error',
 'max_depth': 25,
 'max_features': None,
 'min_samples_split': 7,
 'min_samples_leaf': 7,
 'max_samples': 0.6460792089003213}
```

#### Stacking of Top 2 models with Simple Meta Learner selection and Hyperparameter tuning
Simple meta learner like Linear Regression, KNN and Decision Tree.
Linear Regression as meta learner has given the best performance with average MAE of 3.02.
![alt text](/reports/figures/image-5.png)

#### Final Training of the selected model with best hyperparameters
Linear Regression has no hyperparameters to tune, so not HP tuning of meta learner.
- mae has increased from 2.48 to 3.01
- r2 score has decreased from 0.89 to 0.83
And Avearge of 3 cross validation score is 3.07.

### DVC Pipeline
1. **Data Cleaning**: Load the dataset from the raw data directory. Perform data cleaning and saved the cleaned data into cleaned folder.
2. **data_preparation**: Load the dataset from the cleaned folder and split the dataset into train test dataset. Saved train and test dataset into the interim folder.
3. **data_preprocessing**: Load the train and test datasets from interim folder and perform data preprocessing like encoding and scaling. Then saved the preprocessed train and test datasets into processed folder and save the preprocessor to the models foler.
4. **model training**: Load the training dataset from processed folder and train the model and saving the model to the models folder.
5. **model evaluation**: Load the test dataset from processed folder and evaluate the model and saving the metrics to mlflow.
6. **Register Model**: Register the model for deployment in mlflow model registry with the Aliases `challenger` for testing.

![alt text](/reports/figures/image-1.png)
![alt text](/reports/figures/image-7.png)
![alt text](/reports/figures/image-9.png)
![alt text](/reports/figures/image-10.png)

### FastAPI Development and Testing
We have developed a FastAPI application for the model deployment. It don't need Model Signatures test as FastAPI will handle the input validation using `Pydantic` automatically. It also has built-in Swagger UI for testing the API endpoints. It is fast and Asynchronous. We have tested the API endpoints using Postman.
![alt text](/reports/figures/image.png)

**Input:**
```json
{
  "ID": "0x4607",
  "Delivery_person_ID": "INDORES13DEL02",
  "Delivery_person_Age": "37",
  "Delivery_person_Ratings": "4.9",
  "Restaurant_latitude": 22.745049,
  "Restaurant_longitude": 75.892471,
  "Delivery_location_latitude": 22.765049,
  "Delivery_location_longitude": 75.912471,
  "Order_Date": "19-03-2022",
  "Time_Orderd": "11:30:00",
  "Time_Order_picked": "11:45:00",
  "Weatherconditions": "conditions Sunny",
  "Road_traffic_density": "High",
  "Vehicle_condition": 2,
  "Type_of_order": "Snack",
  "Type_of_vehicle": "motorcycle",
  "multiple_deliveries": "0",
  "Festival": "No",
  "City": "Urban"
}
```
**Output:**
```
18.73
```
add postman api stress testing screenshot

### TODO:
1. add dvc remote.
2. Implement CI/CD pipeline
3. Model signature is not logging on the mlflow model registry. Need to check the issue.

> **NOTE:** `pathlib` is better than `os.path` for path handling and `Joblib` is better than `pickle` for model serialization and deserialization.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```