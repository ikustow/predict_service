import pandas as pd
import numpy as np

# Установка начальной даты и количества месяцев
start_date = '2020-01'
end_date = '2024-06'
date_range = pd.date_range(start=start_date, end=end_date, freq='M')

# Генерация данных для двух продуктов
np.random.seed(42)  # Для воспроизводимости

data = []
products = ['Chocolate', 'Candies']
products = ['Candies']


for product in products:
    for date in date_range:
        orders = np.random.poisson(lam=100) + np.random.randint(-10, 10)  # Случайные колебания заказов
        quantity = np.random.randint(1, 21)  # Количество продуктов в заказе
        data.append([date.strftime('%Y-%m'), product, orders, quantity])

# Создание DataFrame и сохранение в CSV
df = pd.DataFrame(data, columns=['date', 'product', 'orders', 'quantity'])
df.to_csv('csv/historical_orders.csv', index=False)

# Показ первых строк DataFrame
print(df.head())