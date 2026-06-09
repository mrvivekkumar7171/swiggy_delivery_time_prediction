# Swiggy-Delivery-Time-Prediction

Build Ml project that predicts food delivery time in minutes from origin to destination. It is a `Regression` problem that has input features about the `rider`, the `vehicle`
he owns, the `weather` condition, `traffic`, `City`, `Holiday` and `location` of `restaurant` and `delivery`.

### What metrics to use?
We will use RMSE and MAE because we want prediction in minute instead of minute square.

### **How ML Helps in Improving Business**
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

#### TODO:
1. Add dvc remote
2. Add CD pipeline
NOTE: No need to do Model Signatures test as FastAPI will handle the input validation using Pydantic automatically.

> Example of input data for the model:
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

--------

