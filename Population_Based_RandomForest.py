import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

population_data = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_latest.csv")

cases_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")

latest_year = population_data['year'].max()
population_data = population_data[population_data['year'] == latest_year]

cases_data = cases_data.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                            var_name='Date', value_name='Cases')
cases_data['Date'] = pd.to_datetime(cases_data['Date'], errors='coerce')
cases_data = cases_data.groupby(['Country/Region', 'Date']).sum().reset_index()

population_data['entity'] = population_data['entity'].replace({
    "United States": "US",
    "United Kingdom": "UK",
})

data = pd.merge(cases_data, population_data, left_on='Country/Region', right_on='entity', how='inner')

X = data[['population']]
y = data['Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Cases')
plt.plot(y_test.index, predictions, label='Predicted Cases', color='red')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.title('COVID-19 Cases Forecasting Based on Population (Random Forest)')
plt.legend()
plt.show()