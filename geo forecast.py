import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from uszipcode import SearchEngine

"""
# Mock DataFrames
ev_sales = pd.DataFrame({
    'state': ['NY', 'CA', 'IL'],
    'zip': ['12345', '23456', '34567'],
    'registration_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
    'drive_train_type': ['BEV', 'PHEV', 'BEV'],
    'total_sales': [100, 150, 200]  # Assuming aggregation is already done
})

charging_stations = pd.DataFrame({
    'zip': ['12345', '23456', '34567', '89012'],
    'latitude': [40.7128, 34.0522, 41.8781, 36.1699],
    'longitude': [-74.0060, -118.2437, -87.6298, -115.1398],
    'number_of_port_level1': [2, 0, 4, 6],
    'number_of_port_level2': [5, 8, 3, 9],
    'number_of_dc_fast_charger': [3, 2, 0, 0]
})

# Aggregate charging stations data by zip code
charging_stations['total_chargers'] = charging_stations['number_of_port_level1'] + charging_stations['number_of_port_level2'] + charging_stations['number_of_dc_fast_charger']
charging_stations_summary = charging_stations.groupby('zip').agg({
    'latitude': 'first',  # Assuming latitude & longitude are the same for all stations in a zip
    'longitude': 'first',
    'total_chargers': 'sum'
}).reset_index()

# Merge ev_sales with charging_stations_summary to get geographical info
data = pd.merge(ev_sales, charging_stations_summary, on='zip', how='left')

# Handle missing geographical info if any
data.dropna(subset=['latitude', 'longitude'], inplace=True)

# Feature Engineering: Calculate the ratio of sales to chargers
data['sales_to_charger_ratio'] = data['total_sales'] / data['total_chargers']

# Standardize the features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[['latitude', 'longitude', 'sales_to_charger_ratio']])

# Apply KMeans Clustering to identify potential locations
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='viridis', marker='o', s=100, alpha=0.5)

# Plot centroids for new charging station locations
centroids = kmeans.cluster_centers_
centroids[:, :2] = scaler.inverse_transform(centroids)[:, :2]  # Inverse transform lat and lon
plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=50, marker='x')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Locations for New Charging Stations')
plt.grid(True)
plt.show()
"""


ev_data = pd.read_excel('C:\\Users\\BENHAKKI\\Desktop\\Project EVs\\Data\Py Source Code\\Forecasting\\ev-data.xlsx', sheet_name='fc', usecols=['State', 'ZipCounty', 'RegistrationDate', 'Power_Train_Type'])
charging_stations = pd.read_excel('C:\\Users\\BENHAKKI\\Desktop\\Project EVs\\Data\Py Source Code\\Forecasting\\charging-stations.xlsx', sheet_name='cs', usecols=['State', 'City', 'ZIP', 'Longitude',  'Latitude', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count'])

ev_data.rename(columns={'ZipCounty':'zip'}, inplace=True)
charging_stations.rename(columns={'ZIP': 'zip', 'Longitude': 'longitude', 'Latitude': 'latitude'}, inplace=True)

ev_sales_agg = ev_data.groupby('zip').size().reset_index(name='total_sales')
charging_stations['total_chargers'] = charging_stations['EV Level1 EVSE Num'] + charging_stations['EV Level2 EVSE Num'] + charging_stations['EV DC Fast Count']
"""charging_stations_agg = charging_stations.groupby('zip')['total_chargers'].sum().reset_index()
merged_data = pd.merge(ev_sales_agg, charging_stations_agg, on='zip', how='left')
merged_data['sales_to_charger_ratio'] = merged_data['total_sales'] / merged_data['total_chargers']"""

charging_stations_agg = charging_stations.groupby('zip').agg({
    'total_chargers': 'sum',
    'latitude': 'first',  
    'longitude': 'first'
}).reset_index()

merged_data = pd.merge(ev_sales_agg, charging_stations_agg, on='zip', how='left')

merged_data['sales_to_charger_ratio'] = merged_data['total_sales'] / merged_data['total_chargers'].fillna(1)  # Avoid division by zero

features = merged_data[['latitude', 'longitude', 'sales_to_charger_ratio']].fillna(0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42)
merged_data['cluster'] = kmeans.fit_predict(features_scaled)
merged_data['zip'] = merged_data['zip'].apply(lambda x: str(x).zfill(5))

search = SearchEngine()

def get_state_from_zip(zip_code):
    result = search.by_zipcode(zip_code)
    return result.state

merged_data['state'] = merged_data['zip'].apply(get_state_from_zip)

merged_data.to_excel('C:\\Users\\BENHAKKI\\Desktop\\Project EVs\\Data\\Py Source Code\\Forecasting\\merged_data_output.xlsx', index=False)

plt.figure(figsize=(10, 6))
plt.scatter(merged_data['longitude'], merged_data['latitude'], c=merged_data['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Potential Locations for New Charging Stations')
plt.show()





"""
# Assuming 'latitude' and 'longitude' are in your merged_data
features = merged_data[['latitude', 'longitude', 'sales_to_charger_ratio']].fillna(0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42)
merged_data['cluster'] = kmeans.fit_predict(features_scaled)
# Assuming 'merged_data' is your final DataFrame
merged_data.to_excel('C:\\Users\\BENHAKKI\\Desktop\\Project EVs\\Data\\Py Source Code\\Forecasting\\merged_data_output.xlsx', index=False)


plt.figure(figsize=(10, 6))
plt.scatter(merged_data['longitude'], merged_data['latitude'], c=merged_data['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Potential Locations for New Charging Stations')
plt.show()
"""