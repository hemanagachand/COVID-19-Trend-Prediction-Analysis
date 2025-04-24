import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

vaccination_data = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv")

vaccination_data = vaccination_data[['location', 'date', 'total_vaccinations_per_hundred']]
vaccination_data['date'] = pd.to_datetime(vaccination_data['date'], errors='coerce')

cases_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")

cases_data = cases_data.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                            var_name='Date', value_name='Cases')
cases_data['Date'] = pd.to_datetime(cases_data['Date'], errors='coerce')
cases_data = cases_data.groupby(['Country/Region', 'Date']).sum().reset_index()

vaccination_data['location'] = vaccination_data['location'].replace({
    "United States": "US",
    "United Kingdom": "UK",
})

data = pd.merge(cases_data, vaccination_data, left_on=['Country/Region', 'Date'], right_on=['location', 'date'], how='inner')

X = data[['total_vaccinations_per_hundred']]
y = data['Cases']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest
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
plt.title('COVID-19 Cases Forecasting Based on Vaccination Rates (Random Forest)')
plt.legend()
plt.show()