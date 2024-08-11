# Sales Time Series Analysis Using Facebook Prophet

**Author**: Husani Newbold

**Date**: 2024-08-11

## Table of Contents
1. [Introduction & Project Description](#introduction--project-description)
2. [Dataset](#dataset)
3. [Model Structure](#model-structure)
4. [Training the Model](#training-the-model)
5. [Results](#results)
6. [Improvements and Recommendations](#improvements-and-recommendations)
7. [Contributors](#contributors)

## Introduction & Project Description

### Introduction
This project utilizes Facebook's Prophet package, a robust time series forecasting tool, to predict weekly sales data for a Walmart store. By leveraging historical sales data, the goal is to create an accurate forecasting model that can provide insights into future sales trends, aiding in better decision-making for inventory management and strategic planning.

### Project Description
The analysis began with preliminary data cleaning, which involved checking for duplicates and ensuring the dataset's integrity. To create a representative model, we selected a store with the most central sales tendencies, focusing on one that best reflected average sales patterns. After selecting the store, we adjusted the dataset to meet the Prophet model's input requirements, including renaming variables and splitting the data into training and testing sets, with one year of data reserved for model evaluation.

Using the Prophet model, we trained on the historical sales data and achieved highly accurate predictions. The model's performance was validated with a Root Mean Squared Error (RMSE) of $64,000 and a Mean Absolute Percentage Error (MAPE) of 4.5%, demonstrating its effectiveness in forecasting weekly sales with precision.

## Dataset
The dataset was filtered to focus on a single Walmart store by selecting the store whose mean weekly sales were closest to the overall mean of weekly sales across all stores. This approach ensured the selection of a store that exhibited average sales behavior, making it an ideal candidate for modeling typical sales trends.

The resulting dataset spans three years of weekly sales data, from February 5, 2010, to October 26, 2012. This period captures various seasonal trends, sales patterns, and potential anomalies, providing a robust foundation for time series analysis.

<img src="Weekly Sales store 22.png" alt="ANN" width="600" height="300">

The data was then split into training and testing sets. The first two years of data were used to train the Prophet model, offering ample historical information to inform the model. The final 43 weeks of data were reserved as the test set, enabling an evaluation of the model's predictive accuracy and its effectiveness in forecasting future sales trends.

## Model Structure
The Prophet model was configured with specific settings to best capture the underlying patterns in the weekly sales data. The model was initialized with yearly_seasonality enabled and the seasonality_mode set to 'multiplicative'.

Several configurations and additional features of the Prophet model were tested during the analysis. However, this particular configuration proved to be the most effective in capturing the sales patterns. The choice of the multiplicative seasonality mode was particularly crucial. Sales data often exhibits a multiplicative pattern, where seasonal effects and trends scale with the overall level of sales. As demand typically increases over time, the impact of seasonal factors on sales also tends to grow, making a multiplicative model more suitable for this type of data.

This configuration, while simple, allowed the model to accurately account for the varying seasonal effects and produce highly reliable forecasts, as evidenced by the model's strong performance metrics.

```python
# Initialize the Prophet model
model = Prophet(
        yearly_seasonality=True, 
    seasonality_mode='multiplicative'
)
```

## Results

The results of the forecast are visually represented in the chart below. The blue line represents the model's predicted sales, while the red dots indicate the actual sales from the test data. The shaded blue area around the forecast line represents the confidence interval, providing an estimate of the range within which the true values are expected to fall.

<img src="2012 forecast vs actuals .png" alt="ANN" width="600" height="300">

As shown in the chart, the model closely mimics the actual sales data, with almost all of the red points falling within the confidence interval bands. This suggests that the model has effectively captured the underlying patterns and trends in the sales data. The close alignment between the predicted and actual values, with the exception of one outlier, demonstrates the model's accuracy and reliability in forecasting weekly sales. This strong performance is further supported by the low RMSE,MAE & MAPE values, indicating that the model can be confidently used for future sales predictions.

```python
Mean Absolute Error: 45127.23
RMSE: 64428.33
Mape: 4.56
```

## Improvements and Recommendations
- **Incorporate More Data**: The analysis was based on three years of weekly sales data, which provided a solid foundation for the model. However, incorporating a longer historical period could further enhance the model's ability to capture long-term trends and seasonal patterns, leading to even more accurate forecasts.

- **Add More Variables**: The current model only utilized holiday indicators and temperature as additional regressors. Including more relevant variables, such as promotional events, economic indicators, or regional factors, could provide the model with additional context, allowing it to better understand and predict sales fluctuations.

- **Optimize Hyperparameter Tuning**: While the model performed well with the chosen settings, further exploration of hyperparameters, such as adjusting the number of changepoints, changing the seasonality mode, or tweaking the prior scales, could help in fine-tuning the model. Experimenting with these parameters may uncover configurations that provide even better predictive accuracy.

## Contributors
Husani Newbold (Author)



