import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('C:\\Users\\BENHAKKI\\Desktop\\Project EVs\\Data\Py Source Code\\Forecasting\\forecast.xlsx', sheet_name='fc', usecols=['RegistrationDate', 'Power_Train_Type'])

# Convert the 'RegistrationDate' to a datetime format, assuming the format is something like "1 January 2023"
df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], format='%A, %B %d, %Y')

# Check the dataframe
print(df.head())

# Group by RegistrationDate and count the number of Power_Train_Type entries which represent sales
df_sales = df.groupby('RegistrationDate').count().reset_index()
df_sales.columns = ['ds', 'y']
#print(df_sales)
# Initialize the Prophet model
model = Prophet()

# Fit the model with your dataframe
model.fit(df_sales)
# Create a future dataframe for the next 2 years, without assuming a daily frequency since the dates are not continuous
max_date = df_sales['ds'].max()
periods = (max_date + pd.DateOffset(years=2) - max_date).days
future = model.make_future_dataframe(periods=periods)
# Use the model to make predictions
forecast = model.predict(future)
# Plot the forecast
fig = model.plot(forecast)
plt.title('EV Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
# Plot the forecast components
fig2 = model.plot_components(forecast)