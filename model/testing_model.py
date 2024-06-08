import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np


data = pd.read_csv('csv/historical_orders.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
ts = data['orders']

# Обучение модели ARIMA
order = (1, 1, 1)  # Порядок модели ARIMA
model = ARIMA(ts, order=order)
fitted_model = model.fit()

# Прогнозирование на июнь 2024
forecast = fitted_model.forecast(steps=1)
print(f"Прогноз на июнь 2024: {forecast.values[1]}")

# Загрузка данных
data = pd.read_csv('csv/historical_orders.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Создание временного ряда
ts = data['quantity']

# Разделение данных на тренировочный и тестовый наборы
train_size = int(len(ts) * 0.8)
train_data = ts[:train_size]
test_data = ts[train_size:]

# Обучение модели ARIMA
order = (1, 1, 1)  # Порядок модели ARIMA
model = ARIMA(train_data, order=order)
fitted_model = model.fit()

# Прогнозирование
forecast = fitted_model.forecast(steps=len(test_data))

# Оценка модели
mse = mean_squared_error(test_data, forecast)
print(f"Mean Squared Error (MSE) для ARIMA: {mse}")