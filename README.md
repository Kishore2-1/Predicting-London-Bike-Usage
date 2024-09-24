# Predicting London Bike Usage Using Machine Learning

## Overview

This repository contains the code and analysis for predicting bike usage patterns in London using machine learning techniques. The case study focuses on improving the efficiency of the London bike-sharing system (Santander Cycles) by analyzing urban mobility trends and predicting bike demand under various conditions such as weather and time of day.

The project covers multiple aspects, including data collection, cleaning, exploratory data analysis (EDA), geospatial analysis, and predictive modeling using machine learning. It aims to provide insights for better resource allocation and user satisfaction within the bike-sharing system.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Data Collection and Preparation](#data-collection-and-preparation)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Machine Learning Model Building](#machine-learning-model-building)
6. [Key Findings and Recommendations](#key-findings-and-recommendations)
7. [Future Work](#future-work)
8. [Installation and Setup](#installation-and-setup)
9. [References](#references)

---

## Introduction

With cycling being a sustainable form of urban transportation, this project aims to predict the usage patterns of London’s bike-sharing system. By examining factors like time of day, weather conditions, and station popularity, the study develops models to predict demand, optimize bike distribution, and ultimately improve user experience. 

This case study integrates various machine learning techniques to forecast bike demand and make recommendations to enhance the system’s efficiency.

---

## Literature Review

The project draws upon previous research to guide its methodologies, focusing on urban mobility, the factors affecting bike-sharing demand, and the application of machine learning in public transport systems. Machine learning algorithms like Random Forest and XGBoost were explored and compared based on their performance for bike demand prediction.

---

## Data Collection and Preparation

Data used in this project includes:
- **Bike Usage Data**: Downloaded from Transport for London (TfL) for June 2023, capturing detailed trip information (start/end stations, trip duration, bike model, etc.).
- **Weather Data**: Collected from Visual Crossing, providing relevant weather conditions such as temperature, precipitation, and wind speed.

The raw data was cleaned, concatenated, and merged into a unified dataset for analysis. The key steps included handling missing values, formatting timestamps, and filtering to ensure relevant data was used for the study.

---

## Exploratory Data Analysis (EDA)

EDA was conducted to better understand the urban mobility patterns and key factors influencing bike usage:
- **Customer Behavior Analysis**: Examined trends in rental duration and frequency, showing peak periods of usage.
- **Geospatial Analysis**: Identified popular start and end stations as well as the most commonly used routes.
- **Temporal Analysis**: Analyzed bike usage patterns by time of day and day of the week, revealing distinct trends during commute hours and weekends.

Visualizations such as heatmaps, bar plots, and line plots were generated to uncover insights into bike-sharing usage.

---

## Machine Learning Model Building

Several machine learning models were developed to predict bike usage and demand. The project focused on three key tasks:
1. **Resource Allocation Prediction**: Predicting the number of bikes required at each station during peak hours.
2. **Bike Model Prediction**: Identifying which bike models (e.g., classic or electric) will be used based on station and weather conditions.
3. **Hourly Demand Forecasting**: Predicting the number of bike rentals on an hourly basis.

Key machine learning algorithms used include:
- **Ridge Regression**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**

The models were evaluated using metrics like RMSE, R², and accuracy, and their performance was compared to determine the best approach.

---

## Key Findings and Recommendations

From the analysis and machine learning models, several key insights were gained:
- **Peak Demand Periods**: The highest bike usage occurs during weekday commutes (8 AM and 5-6 PM) and weekend afternoons.
- **Weather Influence**: Clear weather and moderate cloud cover see the highest bike usage, while rainy conditions significantly reduce demand.
- **Popular Stations**: Key stations such as **Hyde Park Corner** and **Waterloo Station** consistently see the highest bike traffic.
  
Recommendations for system improvements include:
- Increasing bike availability during peak hours and at high-demand stations.
- Implementing dynamic pricing to encourage off-peak rentals.
- Expanding bike stations in high-demand areas to reduce congestion.

---

## Future Work

Future improvements for the project include:
- **Further Model Tuning**: Fine-tuning machine learning models for higher accuracy.
- **Incorporating More Data**: Expanding the dataset to include additional months or years of bike usage data to capture broader trends.
- **Real-Time Data**: Integrating real-time weather and bike usage data to make more accurate, on-the-fly predictions.
  
---

## Installation and Setup

To run the code in this repository, follow these steps:

1. Clone the repository:
   ```bash
   https://github.com/Kishore2-1/Predicting-London-Bike-Usage.git
   cd Predicting-London-Bike-Usage
   ```

2. Install the required Python libraries:
   ```bash
   pip install pandas
   pip install seaborn
   pip install matplotlib
   pip install windrose
   pip install glob
   pip install pandas numpy scikit-learn xgboost lightgbm tensorflow matplotlib seaborn
   pip install --upgrade tensorflow
   pip install --upgrade scikit-learn
   pip install scikeras
   ```

3. Ensure that the data files (e.g., bike usage data and weather data) are placed in the appropriate directories as specified in the code.

4. Run the Jupyter notebook or Python script to perform the analysis and generate the models.

---

## References

1. Cervero, R., Denman, S., & Jin, Y. (2019). Network design, built and natural environments, and bicycle commuting. *Transport Policy*, 74.
2. Feng, Y., & Wang, S. (2017). A forecast for bicycle rental demand based on random forests and multiple linear regression. *IEEE Conference on Computer and Information Science*.
3. Transport for London (TfL), *Data from the London Cycle Hire Scheme*, June 2023.
4. Visual Crossing, *Weather Data for London*, June 2023.

For a full list of references, see the report included in this repository.

---

This comprehensive analysis and its findings have been documented and made available on GitHub, providing a valuable resource for researchers, policymakers, and practitioners in the field of urban mobility and bike-sharing systems. The repository includes detailed code, data visualizations, and results, promoting transparency and facilitating further research and collaboration.

Feel free to explore the project on GitHub [here](https://github.com/Kishore2-1/Predicting-London-Bike-Usage).
