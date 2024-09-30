# Dengue Prediction using LSTM

This project aims to predict future dengue cases using Long Short-Term Memory (LSTM) neural networks(TensorFlow). The model leverages historical dengue incidence data along with environmental factors such as temperature, humidity, and rainfall to assess their effect on dengue outbreaks and generate accurate predictions. By integrating weather data, the model not only forecasts dengue cases but also quantifies how variations in environmental conditions impact dengue spread.

In addition to LSTM, SARIMAX is used to predict the 2024 dengue cases based on time-series data, ensuring robust future predictions even in the absence of future input data.

## Features
- **LSTM-based time-series prediction** model for dengue outbreak forecasting.
- **SARIMAX-based 2024 prediction** model for time-series forecasting.
- Preprocessing pipeline for **data cleaning** and **normalization**.
- Hyperparameter tuning for **optimal model performance**.
- **Evaluation metrics**: Root Mean Square Error (RMSE), Mean Absolute Error (MAE), R-squared (R²), and Accuracy Score.
- **Visualizations** of training loss, validation loss, and prediction results.

### 2024 actual and predictive data till september 2024 have big difference. meaning this model is not good for predict dengue 
## Visualizations

### Actual vs. Predicted Dengue Cases
![Actual vs Predict Dengue Cases](https://github.com/Juwel2121/Effect-of-weather-on-dengue/blob/main/images/actual%20vs%20predict%20dengue%20case.jpg)
 

### RMSE vs. Epochs (Affected)
![RMSE vs Epochs Affected](https://github.com/Juwel2121/Effect-of-weather-on-dengue/blob/main/images/RMSE%20VS%20Epochs%20affected.jpg)
 

### Comparison of All Models on RMSE
![Comparison All Models on RMSE](https://github.com/Juwel2121/Effect-of-weather-on-dengue/blob/main/images/comparism%20all%20model%20on%20RMSE.jpg)
 

### Training vs. Validation Loss
![Training vs Validation Loss](https://github.com/Juwel2121/Effect-of-weather-on-dengue/blob/main/images/training%20vs%20validation%20loss.jpg)
 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dengue-prediction-lstm.git
   cd dengue-prediction-lstm

 ## Prerequisites

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

