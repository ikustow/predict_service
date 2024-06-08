import pandas as pd
from prophet import Prophet
import json

# Загрузка и подготовка данных
data = pd.read_csv('csv/historical_orders.csv')
data['date'] = pd.to_datetime(data['date'])

# Создание списка уникальных продуктов
products = data['product'].unique()

# Создание пустых датафреймов для хранения прогнозов
order_forecasts = pd.DataFrame()
quantity_forecasts = pd.DataFrame()

# Шаг 2 и 3: Прогнозирование для каждого продукта
for product in products:
    product_data = data[data['product'] == product]
    
    # Прогнозирование количества заказов
    orders_data = product_data[['date', 'orders']].rename(columns={'date': 'ds', 'orders': 'y'})
    order_model = Prophet()
    order_model.fit(orders_data)
    future_orders = order_model.make_future_dataframe(periods=6, freq='M')  # Прогноз на 6 месяцев вперед
    forecast_orders = order_model.predict(future_orders)
    forecast_orders['product'] = product
    order_forecasts = pd.concat([order_forecasts, forecast_orders[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'product']]])
    
    # Прогнозирование количества товаров
    quantity_data = product_data[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
    quantity_model = Prophet()
    quantity_model.fit(quantity_data)
    future_quantity = quantity_model.make_future_dataframe(periods=6, freq='M')  # Прогноз на 6 месяцев вперед
    forecast_quantity = quantity_model.predict(future_quantity)
    forecast_quantity['product'] = product
    quantity_forecasts = pd.concat([quantity_forecasts, forecast_quantity[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'product']]])

# Выбор только новых значений для сохранения в JSON
last_date = data['date'].max()
new_order_forecasts = order_forecasts[order_forecasts['ds'] > last_date]
new_quantity_forecasts = quantity_forecasts[quantity_forecasts['ds'] > last_date]

# Преобразование дат в строковый формат
new_order_forecasts['ds'] = new_order_forecasts['ds'].astype(str)
new_quantity_forecasts['ds'] = new_quantity_forecasts['ds'].astype(str)

# Сохранение прогнозов в файл JSON
order_json = new_order_forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'product']].to_dict(orient='records')
quantity_json = new_quantity_forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'product']].to_dict(orient='records')

with open('order_forecasts.json', 'w') as f:
    json.dump(order_json, f)

with open('quantity_forecasts.json', 'w') as f:
    json.dump(quantity_json, f)
