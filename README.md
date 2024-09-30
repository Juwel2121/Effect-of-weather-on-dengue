# Dengue Prediction using LSTM

This project aims to predict future dengue cases using Long Short-Term Memory (LSTM) neural networks. The model leverages historical dengue incidence data along with environmental factors such as temperature, humidity, and rainfall to assess their effect on dengue outbreaks and generate accurate predictions. By integrating weather data, the model not only forecasts dengue cases but also quantifies how variations in environmental conditions impact dengue spread.


## Features
- LSTM-based time-series prediction model
- Preprocessing pipeline for data cleaning and normalization
- Hyperparameter tuning for optimal model performance
- Model training and evaluation on historical datasets
- Visualizations of training loss and prediction results
- Predict future dengue cases with a trained model



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

