## Analysing Urban Mobility patters and cycling preferences in London’s Diverse population


# Install necessary packages
pip install pandas
pip install seaborn
pip install matplotlib
pip install windrose
pip install glob

# Data Preperation

# Import libraries
import pandas as pd
import seaborn as sns

# Read csv file for June
df1 = pd.read_csv('373JourneyDataExtract05Jun2023-11Jun2023.csv')
df2 = pd.read_csv('374JourneyDataExtract12Jun2023-18Jun2023.csv')
df3 = pd.read_csv('375JourneyDataExtract19Jun2023-30Jun2023.csv')

# Concate the dataframes
df = pd.concat([df1,df2,df3])

# Remove Duplicates values
df = df.drop_duplicates() 
print(df.columns)

# Convert string to datatime 
df['Start date'] = pd.to_datetime(df['Start date'], dayfirst = True)
df['End date'] = pd.to_datetime(df['End date'], dayfirst = True)

# Convert Total duration into Timedelta
df['Total duration'] = pd.to_timedelta(df['Total duration'])

# Filter data for the month of June 2023
df = df[(df['Start date'].dt.month == 6) & (df['Start date'].dt.year == 2023)]

# Resampling (Frequency)
day_resample = df.resample('3D', on = 'Start date').agg({'Total duration': ['sum', 'mean', 'count']})
day_resample.columns = ['Total Duration', 'Average Duration', 'Rental Count']

# Convert durations to hours for easier interpretation
day_resample['Total Duration (hours)'] = day_resample['Total Duration'].dt.total_seconds() / 3600
day_resample['Average Duration (hours)'] = day_resample['Average Duration'].dt.total_seconds() / 3600


# 1. Customer Behaviour Analysis

## a. Rental Duration Analysis

# Import libraries
import matplotlib.pyplot as plt

# Plot total rental duration over time
plt.figure(figsize=(12, 6))
plt.plot(day_resample.index, day_resample['Total Duration (hours)'], label='Total Duration (hours)', color='#196F6B')
plt.title('Total Rental Duration Analysis (3-Day Intervals)')
plt.xlabel('Date resampled for 3 days period')
plt.ylabel('Total Duration (hours)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot average rental duration over time
plt.figure(figsize=(12, 6))
plt.plot(day_resample.index, day_resample['Average Duration (hours)'], label='Average Duration (hours)', color='#FE8402')
plt.title('Average Rental Duration Analysis (3-Day Intervals)')
plt.xlabel('Date resampled for 3 days period')
plt.ylabel('Average Duration (hours)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# b. Frequency Analysis

# Count rentals per user
user_rentals = df.groupby('Bike number').size().reset_index(name='Rental Count')

# Count rentals per station
station_rentals = df.groupby('Start station').size().reset_index(name='Rental Count')

# Temporal analysis - count rentals per day
df['Start date'] = pd.to_datetime(df['Start date'])
daily_rentals = df.set_index('Start date').resample('D').size()

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(daily_rentals.index, daily_rentals.values, color='#196F6B')
plt.title('Daily Rental Frequency')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.show()


## c. Time series Analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Perform Augmented Dickey-Fuller test for stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

test_stationarity(daily_rentals)

# Seasonal decomposition
decomposition = seasonal_decompose(daily_rentals, model='additive', period = 7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot decomposition
plt.figure(figsize=(12, 10))

# Original Plot
plt.subplot(411)
plt.plot(daily_rentals, label='Original', color='#FFDD00')
plt.legend(loc='best')
plt.title('Original')

# Trend Plot
plt.subplot(412)
plt.plot(trend, label='Trend', color='#FE8402')
plt.legend(loc='best')
plt.title('Trend')

# Seasonal Plot
plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color='#6BCAE2')
plt.legend(loc='best')
plt.title('Seasonality')

# Residual Plot
plt.subplot(414)
plt.plot(residual, label='Residuals', color='#41924B')
plt.legend(loc='best')
plt.title('Residuals')

plt.tight_layout()
plt.show()

# ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(daily_rentals, ax=ax1, lags=12)  # Adjusted lags to 20
plot_pacf(daily_rentals, ax=ax2, lags=12)  # Adjusted lags to 20
plt.show()


## 3. Geospatial Analysis

# Popular Routes For Start Station, End Station and Combined Routes

# Count of start stations
start_station_counts = df.groupby(['Start station']).size().sort_values(ascending=False).head(5)

# Count of End stations
end_station_counts = df.groupby(['End station']).size().sort_values(ascending=False).head(5)

# Popular Top 5 Routes From Start to End Station
popular_routes = df.groupby(['Start station', 'End station']).size().sort_values(ascending=False).head(5)

# Print Top 5 Start Station routes
print("Top 5 Most Popular Routes:")
print(start_station_counts)

print(" ")
# Print Top 5 End Station routes
print("Top 5 Most Popular Routes:")
print(end_station_counts)

print(" ")
# Print Top 5 Popular combined routes
print("Top 5 Most Popular Routes:")
print(popular_routes)

# Define a function to get the colors
def get_colors(n_colors):
    return sns.color_palette("YlOrRd", n_colors=n_colors)

# Plot Popular top 5 routes for start stations
plt.figure(figsize=(12, 6))
start_colors = get_colors(len(start_station_counts))
start_station_counts.plot(kind='bar', color=start_colors)
plt.title('Top 5 Most Popular Start Stations')
plt.xlabel('Station Names')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot Popular top 5 routes for end stations
plt.figure(figsize=(12, 6))
end_colors = get_colors(len(end_station_counts))
end_station_counts.plot(kind='bar', color=end_colors)
plt.title('Top 5 Most Popular End Stations')
plt.xlabel('Station Name')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot Popular top 5 routes
plt.figure(figsize=(8, 4))
route_colors = get_colors(len(popular_routes))
popular_routes.plot(kind='bar', color=route_colors)
plt.title('Top 5 Most Popular Routes')
plt.xlabel('Route')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


## 4. Multivarient Analysis

## Trip Duration vs. Time of Day:

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with the necessary columns

# Convert 'Start date' to datetime
df['Start date'] = pd.to_datetime(df['Start date'])

# Extract hour and day of week
df['Start Hour'] = df['Start date'].dt.hour
df['Day of Week'] = df['Start date'].dt.day_name()

# Convert duration to minutes
df['Duration_minutes'] = df['Total duration'].dt.total_seconds() / 60

# Set up the plot
plt.figure(figsize=(20, 10))

# Create the strip plot
sns.stripplot(x='Start Hour', y='Duration_minutes', hue='Day of Week', data=df,
              jitter=True, alpha=0.6, size=8,
              order=range(24),
              hue_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Customize the plot
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Trip Duration (minutes)', fontsize=14)
plt.title('Trip Duration by Hour of Day and Day of Week', fontsize=16)
plt.xticks(range(24), fontsize=12)
plt.yticks(fontsize=12)

# Adjust y-axis to focus on the main part of the distribution
plt.ylim(0, df['Duration_minutes'].quantile(1))

# Move the legend outside the plot
plt.legend(title='Day of Week', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()


## Heat Map

# Convert date columns to datetime
df['Start date'] = pd.to_datetime(df['Start date'])
df['End date'] = pd.to_datetime(df['End date'])

# Define day order for consistent plotting
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Trip Duration by Hour of Day
pivot_duration_hour = df.pivot_table(values='Duration_minutes', index='Start Hour', aggfunc='count')

# Trip Duration by Day of the Week
pivot_duration_day = df.pivot_table(values='Duration_minutes', index='Day of Week', aggfunc='count')
pivot_duration_day = pivot_duration_day.reindex(day_order)  # Reorder days

# Heatmap for Trip Duration by Hour of Day
plt.figure(figsize=(18, 9))
sns.heatmap(pivot_duration_hour, cmap='YlOrRd', annot=True, fmt='.2f')
plt.title('Trip Duration by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('')
plt.show()

# Heatmap for Trip Duration by Day of the Week
plt.figure(figsize=(18, 9))
sns.heatmap(pivot_duration_day, cmap='YlOrRd', annot=True, fmt='.2f')
plt.title('Trip Duration by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('')
plt.show()

# Trip Duration by Hourly Rentals by Day of Week
hourly_rentals = df.pivot_table(values='Number', index= 'Start Hour', columns= 'Day of Week', aggfunc='count')
hourly_rentals = hourly_rentals.reindex(columns=day_order)

# Heatmap for Trip Duration by Hourly Rentals by Day of Week
plt.figure(figsize=(12, 6))
sns.heatmap(hourly_rentals, cmap='YlOrRd', annot=False, fmt='d')
plt.title('Hourly Rentals by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Hour of Day')
plt.tight_layout()
plt.show()



## Station Popularity vs. Day of Week

# Extract day of the week
df['Day of Week'] = df['Start date'].dt.day_name()

# Define the order of days for consistent plotting
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Top 10 Most Popular Start Stations
top_10_stations = df['Start station'].value_counts().nlargest(10).index

print(top_10_stations)

# Create a pivot table for start station popularity by day of week
station_popularity = df[df['Start station'].isin(top_10_stations)].pivot_table(
    values='Number',
    index='Start station',
    columns='Day of Week',
    aggfunc='count',
    fill_value=0
)

# Reorder columns based on day_order
station_popularity = station_popularity.reindex(columns=day_order)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(station_popularity, cmap='YlOrRd', annot=False, fmt='d')
plt.title('Top 10 Start Stations: Usage by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Start Station')
plt.tight_layout()
plt.show()

# Normalized Station Popularity (percentage of weekly usage)
station_popularity_normalized = station_popularity.div(station_popularity.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 8))
sns.heatmap(station_popularity_normalized, cmap='YlOrRd', annot=False, fmt='.1f')
plt.title('Top 10 Start Stations: Normalized Usage by Day of Week (%)')
plt.xlabel('Day of Week')
plt.ylabel('Start Station')
plt.tight_layout()
plt.show()

# Line plot to show daily trends for each station
plt.figure(figsize=(12, 6))
for station in top_10_stations:
    plt.plot(day_order, station_popularity.loc[station], marker='o', label=station)
plt.title('Daily Usage Trends for Top 10 Start Stations')
plt.xlabel('Day of Week')
plt.ylabel('Number of Trips')
plt.legend(title='Start Station', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()






### Predictive Model Using Machine Learning


# Install necessary packages
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow matplotlib seaborn
pip install --upgrade tensorflow
pip install --upgrade scikit-learn
pip install scikeras

# Data Preparation

## Loading Data Bike usage Dataset June 2023

# Import libraries
import pandas as pd
import seaborn as sns

# Read csv file for Bike Usage June 2023 Dataset
df1 = pd.read_csv('373JourneyDataExtract05Jun2023-11Jun2023.csv')
df2 = pd.read_csv('374JourneyDataExtract12Jun2023-18Jun2023.csv')
df3 = pd.read_csv('375JourneyDataExtract19Jun2023-30Jun2023.csv')

# Concate the dataframes
df = pd.concat([df1,df2,df3])

# Remove Duplicates values
df = df.drop_duplicates() 
print(df.columns)

# Convert string to datatime 
df['Start date'] = pd.to_datetime(df['Start date'], dayfirst = True)
df['End date'] = pd.to_datetime(df['End date'], dayfirst = True)

# Convert Total duration into Timedelta
df['Total duration'] = pd.to_timedelta(df['Total duration'])

# convert Total duration(ms) to minutes
df['Total duration (min)'] = df['Total duration (ms)'] / 60000


## Loading Data Weather Dataset For June 2023

# Import libraries
import pandas as pd

# Read Weather Datasets csv file for June
weather_data = pd.read_csv('Weather_data_June.csv')

# Select relevant columns from weather data
relevant_weather_columns = ['datetime', 'temp', 'precip', 'humidity', 
                            'windspeed', 'cloudcover', 'visibility', 'conditions',
                            'solarradiation', 'uvindex']

# Update relevant columns
weather_data = weather_data[relevant_weather_columns]

# Convert date columns to datetime
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])

# Set datetime as index for weather_data
weather_data.set_index('datetime', inplace=True)



## Data Cleaning

# Split 'Start station' into 'Start Station' and 'Start City'
split_columns = df['Start station'].str.split(',', n=1, expand=True)

# Assign the new columns to the DataFrame, handling cases with missing values
df['Start Station'] = split_columns[0].str.strip()
df['Start City'] = split_columns[1].str.strip()

# Replace NaN values in 'Start City' with a placeholder or keep them as NaN
df['Start City'] = df['Start City'].fillna('Unknown')

# Split 'End station' into 'Destination Station' and 'Destination City'
split_columns = df['End station'].str.split(',', n=1, expand=True)

# Assign the new columns to the DataFrame, handling cases with missing values
df['Destination Station'] = split_columns[0].str.strip()
df['Destination City'] = split_columns[1].str.strip()

# Replace NaN values in 'Start City' with a placeholder or keep them as NaN
df['Destination City'] = df['Destination City'].fillna('Unknown')

# Select relevant columns from Bike Usge data
relevant_Bike_usage_columns = ['Start date', 'Start Station',
       'Destination Station',
       'End date',
       'Bike model', 'Total duration (min)']
df = df[relevant_Bike_usage_columns]

# Selecting Station as "Hyde Park Corner" For Analysis

# Filter for top stations
df_top_stations = ['Hyde Park Corner']
df_top_stations = df[df['Start Station'].isin(df_top_stations)]

# Round bike start times to nearest hour
df_top_stations['Rounded_Start_Time'] = df_top_stations['Start date'].dt.round('H')

# Merge datasets
merged_data = pd.merge_asof(df_top_stations.sort_values('Rounded_Start_Time'),
                            weather_data,
                            left_on='Rounded_Start_Time',
                            right_index=True,
                            direction='nearest')

# Drop temporary column
merged_data.drop('Rounded_Start_Time', axis=1, inplace=True)



# Exploratory Data Analysis(EDA)

## Frequency of Top 10 Destinantion Stations for Different Bike Models

import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'Bike model' and 'Destination Station' and count the occurrences
grouped = merged_data.groupby(['Bike model', 'Destination Station']).size().reset_index(name='Count')

# Plot for each bike model
bike_models = merged_data['Bike model'].unique()
for bike_model in bike_models:
    # Filter data for the current bike model
    model_data = grouped[grouped['Bike model'] == bike_model]
    
    # Get the top 10 destination stations
    top_destinations = model_data.nlargest(10, 'Count')
    
    # Plotting
    plt.figure(figsize=(14, 7))
    ax = top_destinations.sort_values('Destination Station').plot(kind='bar', x='Destination Station', y='Count', legend=False, color=sns.color_palette("YlGnBu", n_colors=10))
    plt.xlabel('Destination Station', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title(f'Top 10 Destination Stations for {bike_model}', fontsize=10)
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(fontsize=9)
    plt.show()

    
## Trip Frequency by Hour of the Day and Day of the Week

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract hour of the day and day name from 'Start date'
merged_data['Hour'] = merged_data['Start date'].dt.hour
merged_data['Day'] = merged_data['Start date'].dt.day_name()

# Create a pivot table to count the number of trips for each combination of hour and day
pivot_table = merged_data.pivot_table(index='Hour', columns='Day', values='Start Station', aggfunc='count').fillna(0)

# Reorder the days to start from Monday
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_table = pivot_table[days_order]

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=False, fmt=".0f", cmap="YlGnBu")
plt.title('Heatmap of Trip Frequency by Hour of the Day and Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Hour of the Day')
plt.show()


## Trip Frequency For Different Weather Conditions

# Count the occurrences of each condition
condition_counts = merged_data['conditions'].value_counts()

# Generate color palette
colors = sns.color_palette("YlGnBu", n_colors=len(condition_counts))

# Plot the frequency of different conditions
plt.figure(figsize=(10, 6))
condition_counts.plot(kind='bar', color=colors)
plt.xlabel('Condition', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Trip Frequency For Different Weather Conditions', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# Duplicate the DataFrame
merged_d = merged_data.copy()



## Scatter plot

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Convert categorical columns to numerical values using LabelEncoder
label_encoders = {}
categorical_columns = ['Start Station', 'Destination Station', 'Bike model', 'conditions']

for column in categorical_columns:
    if column in merged_d.columns:
        le = LabelEncoder()
        merged_d[column] = le.fit_transform(merged_d[column].astype(str))
        label_encoders[column] = le

# Ensure all columns to plot are in numerical format
columns_to_plot = ['Total duration (min)', 'temp', 'precip', 'humidity',
                   'windspeed', 'cloudcover', 'visibility', 'solarradiation', 'uvindex',
                   'Destination Station', 'Bike model', 'conditions']

# Verify and convert to numerical if necessary
merged_d[columns_to_plot] = merged_d[columns_to_plot].apply(pd.to_numeric, errors='coerce')

# Remove any rows with NaN values resulting from conversion
merged_d.dropna(subset=columns_to_plot, inplace=True)

# Draw scatter plot matrix using seaborn's pairplot
sns.pairplot(merged_d[columns_to_plot], diag_kind='kde')
plt.suptitle('Scatter Plot Matrix for EDA', y=1.02)
plt.show()



## Heatmap

# Calculate the correlation matrix
correlation_matrix = merged_d[columns_to_plot].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='YlGnBu', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Duplicate the DataFrame
merged_data_1 = merged_data.copy()


## Feature Engineering

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. Extract relevant time features
merged_data_1['Hour'] = merged_data_1['Start date'].dt.hour
merged_data_1['Day of week'] = merged_data_1['Start date'].dt.day_name()
merged_data_1['Is weekend'] = merged_data_1['Day of week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Time of day categories
merged_data_1['Time of day'] = merged_data_1['Hour'].apply(lambda x: 'morning' if 5 <= x < 12 else 'afternoon' if 12 <= x < 17 else 'evening' if 17 <= x < 21 else 'night')

# 2. Temperature-related features
merged_data_1['apparent_temp'] = merged_data_1['temp'] + 0.33 * merged_data_1['humidity'] - 0.70 * merged_data_1['windspeed'] - 4.00

# 3. Solar radiation and UV index features
merged_data_1['solar_uv_interaction'] = merged_data_1['solarradiation'] * merged_data_1['uvindex']

# 4. Cloud cover and visibility features
merged_data_1['clear_sky'] = ((merged_data_1['cloudcover'] < 20) & (merged_data_1['visibility'] > 10)).astype(int)

# 5. Humidity-related features
merged_data_1['humidity_category'] = pd.cut(merged_data_1['humidity'], bins=[0, 30, 60, 100], labels=['Dry', 'Comfortable', 'Humid'])

# 6. Composite weather features
# Adjust this value based on the desired impact of precipitation
alpha = 1 
merged_data_1['weather_comfort_index'] = (merged_data_1['temp']- 0.55 * (1 - merged_data_1['humidity'] / 100) * (merged_data_1['temp'] - 14.5)
    - 0.2 * merged_data_1['windspeed'] + 0.1 * merged_data_1['cloudcover'] - alpha * merged_data_1['precip'])

# 7. Categorize Trip Duration
# Ensure 'Total duration (min)' is in numerical format
merged_data_1['Total duration (min)'] = pd.to_numeric(merged_data_1['Total duration (min)'], errors='coerce')

bins = [0, 5, 15, 30, 60, 120, np.inf]
labels = ['<5 min', '5-15 min', '15-30 min', '30-60 min', '1-2 hours', '>2 hours']
merged_data_1['trip_duration_category'] = pd.cut(merged_data_1['Total duration (min)'], bins=bins, labels=labels)

# 8. Interaction with Time of Day and Day of Week
merged_data_1['duration_time_of_day'] = merged_data_1.groupby('Time of day')['Total duration (min)'].transform('mean')
merged_data_1['duration_day_of_week'] = merged_data_1.groupby('Day of week')['Total duration (min)'].transform('mean')

# Ensure all columns to plot are in numerical format
columns_to_plot = ['Total duration (min)','Destination Station', 'Bike model', 'conditions',
                   'Hour', 'Is weekend', 'Time of day', 'Day of week', 'apparent_temp', 'solar_uv_interaction', 
                   'clear_sky', 'humidity_category', 'weather_comfort_index','duration_day_of_week','duration_time_of_day','trip_duration_category']

merged_data_2 = merged_data_1[columns_to_plot]

# Convert categorical columns to numerical values using LabelEncoder
label_encoders = {}
categorical_columns = ['Destination Station', 'Bike model','Time of day','duration_time_of_day','conditions', 'trip_duration_category', 'Day of week', 'humidity_category']

for column in categorical_columns:
    if column in merged_data_1.columns:
        le = LabelEncoder()
        merged_data_1[column] = le.fit_transform(merged_data_1[column].astype(str))
        label_encoders[column] = le

# Verify and convert to numerical if necessary
merged_data_1[columns_to_plot] = merged_data_1[columns_to_plot].apply(pd.to_numeric, errors='coerce')

# Keep only relevent columns
merged_data_1 = merged_data_1[columns_to_plot]

# Remove any rows with NaN values resulting from conversion
merged_data_1.dropna(subset=columns_to_plot, inplace=True)

# Calculate the correlation matrix
correlation_matrix = merged_data_1[columns_to_plot].corr()


# Plot the heatmap for New Features
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='YlGnBu', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Duplicate the DataFrame
merged_data = merged_data_2.copy()


# Improving resource allocation.

# Import necessary libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Data Preparation

# Calculate the number of trips starting and ending at each station for each hour
demand_data = merged_data.groupby(['Destination Station', 'Hour', 'Day of week', 'Is weekend', 'Time of day']).size().reset_index(name='trip_count')

# Merge with original data to get features
merged_data = merged_data.merge(demand_data, on=['Destination Station', 'Hour', 'Day of week', 'Is weekend', 'Time of day'], how='left')

# Feature Engineering

# Define the features and target
features = ['Hour', 'Day of week', 'Is weekend', 'Time of day', 'conditions', 'humidity_category', 'weather_comfort_index']
target = 'trip_count'

# Define categorical and numerical columns
categorical_columns = ['Day of week', 'Is weekend', 'Time of day', 'conditions', 'humidity_category']
numerical_columns = ['Hour', 'weather_comfort_index']

# Fill numerical columns with the mean
for column in numerical_columns:
    merged_data[column].fillna(merged_data[column].mean(), inplace=True)

# Fill categorical columns with the mode
for column in categorical_columns:
    merged_data[column].fillna(merged_data[column].mode()[0], inplace=True)

# Encode categorical variables using ColumnTransformer and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Split the data into training and testing sets
X = merged_data[features]
y = merged_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipelines dictionary with regression models
pipelines = {
    'Ridge Regression': Pipeline(steps=[('preprocessor', preprocessor), ('model', Ridge())]),
    'Lasso Regression': Pipeline(steps=[('preprocessor', preprocessor), ('model', Lasso())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=42))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(random_state=42))]),
    'LightGBM': Pipeline(steps=[('preprocessor', preprocessor), ('model', LGBMRegressor(random_state=42))]),
}

# Define hyperparameter grids for regression models
param_grids = {
    'Ridge Regression': {'model__alpha': [0.1, 1.0, 10.0]},
    'Lasso Regression': {'model__alpha': [0.1, 1.0, 10.0]},
    'Decision Tree': {'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Gradient Boosting': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'XGBoost': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'LightGBM': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
}

# Cross Validation using Randomized Search

# Evaluate models with Randomized Search
best_models = {}
rmse_results = {}  # Initialize RMSE results dictionary
r2_results = {}  # Initialize R^2 results dictionary
time_results = {}  # Initialize time results dictionary

for name, pipeline in pipelines.items():
    start_time = time.time()  # Start timer
    if name in param_grids:
        # Reduce n_iter to a smaller number (e.g., 10) to speed up the search process
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=None, verbose=2, random_state=42)
    else:
        search = pipeline
    search.fit(X_train, y_train)
    end_time = time.time()  # End timer
    best_models[name] = search.best_estimator_ if name in param_grids else search
    y_pred = search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    rmse_results[name] = rmse  # Store RMSE in the dictionary
    r2_results[name] = r2  # Store R^2 in the dictionary
    time_taken = end_time - start_time  # Calculate time taken
    time_results[name] = time_taken  # Store time taken in the dictionary
    print(f'{name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}, Time Taken: {time_taken:.2f} seconds')

# Print RMSE results
for name, rmse in rmse_results.items():  # Iterate over RMSE results dictionary
    print(f'{name} - RMSE: {rmse:.2f}')

# Print R^2 results
for name, r2 in r2_results.items():  # Iterate over R^2 results dictionary
    print(f'{name} - R^2: {r2:.2f}')

# Print time results
for name, time_taken in time_results.items():  # Iterate over time results dictionary
    print(f'{name} - Time Taken: {time_taken:.2f} seconds')




## Predict Bike Model

# Import necessary libraries
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier


# Data Preparation

# Define the features and target
features = ['Destination Station', 'conditions', 'Time of day', 'weather_comfort_index']
target = 'Bike model'

# Define categorical and numerical columns
categorical_columns = ['Destination Station', 'conditions', 'Time of day']
numerical_columns = ['weather_comfort_index']

# Fill numerical columns with the mean
for column in numerical_columns:
    merged_data[column].fillna(merged_data[column].mean(), inplace=True)

# Fill categorical columns with the mode
for column in categorical_columns:
    merged_data[column].fillna(merged_data[column].mode()[0], inplace=True)

# Encode categorical variables using ColumnTransformer and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Encode the target variable 'Bike model'
label_encoder = LabelEncoder()
merged_data[target] = label_encoder.fit_transform(merged_data[target])

# Split the data into training and testing sets
X = merged_data[features]
y = merged_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipelines dictionary with classification models only
pipelines = {
    'Ridge Classifier': Pipeline(steps=[('preprocessor', preprocessor), ('model', RidgeClassifier())]),
    'Logistic Regression (Lasso)': Pipeline(steps=[('preprocessor', preprocessor), ('model', LogisticRegression(penalty='l1', solver='liblinear'))]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeClassifier(random_state=42))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier(random_state=42))]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingClassifier(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBClassifier(random_state=42))]),
    'LightGBM': Pipeline(steps=[('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42))]),
}

# Define hyperparameter grids for classification models
param_grids = {
    'Ridge Classifier': {'model__alpha': [0.1, 1.0, 10.0]},
    'Logistic Regression (Lasso)': {'model__C': [0.1, 1.0, 10.0]},
    'Decision Tree': {'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Gradient Boosting': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'XGBoost': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'LightGBM': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
}

# Cross Validation using Randomized Search

# Evaluate models with Randomized Search
best_models = {}

 # Initialize accuracy_results dictionary
accuracy_results = {} 

# Initialize time_results dictionary
time_results = {}  

for name, pipeline in pipelines.items():
    start_time = time.time()  # Start timer
    if name in param_grids:
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], n_iter=10, cv=3, scoring='accuracy', n_jobs=None, verbose=2, random_state=42)  # Changed n_jobs to None
    else:
        search = pipeline
    search.fit(X_train, y_train)
    end_time = time.time()  # End timer
    best_models[name] = search.best_estimator_ if name in param_grids else search
    y_pred = search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[name] = accuracy  # Store accuracy in the dictionary
    time_taken = end_time - start_time  # Calculate time taken
    time_results[name] = time_taken  # Store time taken in the dictionary
    print(f'{name} - Accuracy: {accuracy}, Time Taken: {time_taken:.2f} seconds')

# Print accuracy results
for name, accuracy in accuracy_results.items():  # Iterate over accuracy_results dictionary
    print(f'{name} - Accuracy: {accuracy}')

# Print time results
for name, time_taken in time_results.items():  # Iterate over time_results dictionary
    print(f'{name} - Time Taken: {time_taken:.2f} seconds')




# Predicting Hourly Demand

# Import necessary libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Data Preparation

# Aggregate data to create demand metrics 
hourly_demand = merged_data.groupby(['Destination Station','Hour']).size().reset_index(name='demand')

# Merge with original data to get features
merged_data = merged_data.merge(hourly_demand, on=['Destination Station', 'Hour'], how='left')

# Define the features and target
features = ['Destination Station', 'Bike model', 'conditions', 'Hour', 'Time of day', 'solar_uv_interaction', 'clear_sky', 'humidity_category', 'weather_comfort_index', 'trip_duration_category']
target = 'demand'

# Define categorical and numerical columns
categorical_columns = ['Destination Station', 'Bike model', 'conditions', 'Time of day', 'humidity_category', 'trip_duration_category']
numerical_columns = ['Hour','solar_uv_interaction', 'clear_sky', 'weather_comfort_index']

# Fill numerical columns with the mean
for column in numerical_columns:
    merged_data[column].fillna(merged_data[column].mean(), inplace=True)

# Fill categorical columns with the mode
for column in categorical_columns:
    merged_data[column].fillna(merged_data[column].mode()[0], inplace=True)

# Encode categorical variables using ColumnTransformer and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Split the data into training and testing sets
X = merged_data[features]
y = merged_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipelines dictionary with regression models
pipelines = {
    'Ridge Regression': Pipeline(steps=[('preprocessor', preprocessor), ('model', Ridge())]),
    'Lasso Regression': Pipeline(steps=[('preprocessor', preprocessor), ('model', Lasso())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=42))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(random_state=42))]),
    'LightGBM': Pipeline(steps=[('preprocessor', preprocessor), ('model', LGBMRegressor(random_state=42))]),
}

# Define hyperparameter grids for regression models
param_grids = {
    'Ridge Regression': {'model__alpha': [0.1, 1.0, 10.0]},
    'Lasso Regression': {'model__alpha': [0.1, 1.0, 10.0]},
    'Decision Tree': {'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Gradient Boosting': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'XGBoost': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'LightGBM': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
}

# Cross Validation using Randomized Search

# Evaluate models with Randomized Search
best_models = {}
rmse_results = {}  # Initialize RMSE results dictionary
r2_results = {}  # Initialize R^2 results dictionary
time_results = {}  # Initialize time results dictionary

for name, pipeline in pipelines.items():
    start_time = time.time()  # Start timer
    if name in param_grids:
        # Reduce n_iter to a smaller number (e.g., 10) to speed up the search process
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=None, verbose=2, random_state=42)
    else:
        search = pipeline
    search.fit(X_train, y_train)
    end_time = time.time()  # End timer
    best_models[name] = search.best_estimator_ if name in param_grids else search
    y_pred = search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    rmse_results[name] = rmse  # Store RMSE in the dictionary
    r2_results[name] = r2  # Store R^2 in the dictionary
    time_taken = end_time - start_time  # Calculate time taken
    time_results[name] = time_taken  # Store time taken in the dictionary
    print(f'{name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}, Time Taken: {time_taken:.2f} seconds')

# Print RMSE results
for name, rmse in rmse_results.items():  # Iterate over RMSE results dictionary
    print(f'{name} - RMSE: {rmse:.2f}')

# Print R^2 results
for name, r2 in r2_results.items():  # Iterate over R^2 results dictionary
    print(f'{name} - R^2: {r2:.2f}')

# Print time results
for name, time_taken in time_results.items():  # Iterate over time results dictionary
    print(f'{name} - Time Taken: {time_taken:.2f} seconds')





## Predict Total Duration


# Import necessary libraries
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Data Preparation

# Define the features and target
features = ['Destination Station', 'Bike model', 'conditions', 'Hour', 'Time of day', 'solar_uv_interaction', 'clear_sky', 'humidity_category', 'weather_comfort_index', 'trip_duration_category']
target = 'Total duration (min)'

# Define categorical and numerical columns
categorical_columns = ['Destination Station', 'Bike model', 'conditions', 'Time of day', 'humidity_category', 'trip_duration_category']
numerical_columns = ['Hour','solar_uv_interaction', 'clear_sky', 'weather_comfort_index']

# Fill numerical columns with the mean
for column in numerical_columns:
    merged_data[column].fillna(merged_data[column].mean(), inplace=True)

# Fill categorical columns with the mode
for column in categorical_columns:
    merged_data[column].fillna(merged_data[column].mode()[0], inplace=True)

# categorical variables using ColumnTransformer and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Split the data into training and testing sets
X = merged_data[features]
y = merged_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipelines dictionary with regression models
pipelines = {
    'Ridge Regression': Pipeline(steps=[('preprocessor', preprocessor), ('model', Ridge())]),
    'Lasso Regression': Pipeline(steps=[('preprocessor', preprocessor), ('model', Lasso())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=42))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(random_state=42))]),
    'LightGBM': Pipeline(steps=[('preprocessor', preprocessor), ('model', LGBMRegressor(random_state=42))]),
}

# Define hyperparameter grids for regression models
param_grids = {
    'Ridge Regression': {'model__alpha': [0.1, 1.0, 10.0]},
    'Lasso Regression': {'model__alpha': [0.1, 1.0, 10.0]},
    'Decision Tree': {'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Gradient Boosting': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'XGBoost': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
    'LightGBM': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.01, 0.1, 0.2], 'model__max_depth': [3, 5, 7]},
}

# Cross Validation using Randomized Search

# Evaluate models with Randomized Search
best_models = {}

# Initialize RMSE results dictionary
rmse_results = {}  

# Initialize R^2 results dictionary
r2_results = {}  

# Initialize time results dictionary
time_results = {}  

for name, pipeline in pipelines.items():
    start_time = time.time()  # Start timer
    if name in param_grids:
        search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=None, verbose=2, random_state=42)
    else:
        search = pipeline
    search.fit(X_train, y_train)
    end_time = time.time()  # End timer
    best_models[name] = search.best_estimator_ if name in param_grids else search
    y_pred = search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    rmse_results[name] = rmse  # Store RMSE in the dictionary
    r2_results[name] = r2  # Store R^2 in the dictionary
    time_taken = end_time - start_time  # Calculate time taken
    time_results[name] = time_taken  # Store time taken in the dictionary
    print(f'{name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}, Time Taken: {time_taken:.2f} seconds')

# Print RMSE results
for name, rmse in rmse_results.items():  # Iterate over RMSE results dictionary
    print(f'{name} - RMSE: {rmse:.2f}')

# Print R^2 results
for name, r2 in r2_results.items():  # Iterate over R^2 results dictionary
    print(f'{name} - R^2: {r2:.2f}')

# Print time results
for name, time_taken in time_results.items():  # Iterate over time results dictionary
    print(f'{name} - Time Taken: {time_taken:.2f} seconds')


