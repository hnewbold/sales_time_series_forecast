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
This project utilizes Facebook's Prophet package, a robust time series forecasting tool, to predict weekly sales data for a Walmart store. By leveraging historical sales data and a few key independent variables, the model is able to provide highly accurate forecasts of future sales trends which can aid in better decision-making for strategic planning.

### Project Description
The analysis began with basic data cleaning, which included checking for duplicates and ensuring that the weekly sales values were complete and continuous. The store with the most central sales tendencies was identified and selected for this analysis.

Once the data was cleaned and preprocessed, the Prophet model was tuned and trained, testing various hyperparameters. In the end, the model performed exceptionally well, achieving a validated Root Mean Squared Error (RMSE) of $64,000 and a Mean Absolute Percentage Error (MAPE) of 4.5%. 

## Dataset
The dataset was filtered to focus on a single Walmart store by selecting the one whose mean weekly sales were closest to the overall mean of weekly sales across all stores. This approach was used in this first iteration of the model to test it on the store that exhibited the most typical sales behavior. The decision not to use every store was made to avoid potential noise in the model caused by differences in store sizes and locations. In the future, a more nuanced approach would be needed to forecast sales across the entire population of stores.

The resulting dataset spans three years of weekly sales data, from February 5, 2010, to October 26, 2012. As shown below, this period captures various seasonal trends, sales patterns, and potential anomalies, providing a robust foundation for time series analysis.

<img src="Weekly Sales store 22.png" alt="ANN" width="600" height="300">

The data was then split into training and testing sets. The first two years of data were used to train the Prophet model, offering ample historical information to inform the model. The final 43 weeks of data were reserved as the test set, enabling an evaluation of the model's predictive accuracy and its effectiveness in forecasting future sales trends.

## Model Structure
The Prophet model was configured with specific settings to best capture the underlying patterns in the weekly sales data. The model was initialized with yearly_seasonality enabled and the seasonality_mode set to 'multiplicative'.

Several configurations and additional features of the Prophet model were tested during the analysis. However, this particular configuration proved to be the most effective in capturing the sales patterns. Using the multiplicative seasonality_mode as opposed to additive seemed to make little difference, which is surprising given that sales data often exhibits a multiplicative pattern, where seasonal effects and trends scale with the overall level of sales. The inclusion of yearly_seasonality was however crucial, significantly improving model performance. This improvement does make sense though given that Walmart's sales are likely to be influenced by strong annual patterns, such as holiday shopping seasons and other recurring events that have a substantial impact on consumer behavior year after year.

```python
# Initialize the Prophet model
model = Prophet(
        yearly_seasonality=True, 
    seasonality_mode='multiplicative'
)
```

## Results

In the graph below, the model's forecast is compared to the actual sales values. The blue line represents the model's predicted sales, while the red dots indicate the actual sales from the test data. The shaded blue area around the forecast line represents the confidence interval, providing an estimate of the range within which the true values are expected to fall.

<img src="2012 forecast vs actuals .png" alt="ANN" width="600" height="300">

As shown in the chart, the model's predictions closely align with the actual sales data. Almost all of the red points fall within the confidence interval bands, suggesting that the model has effectively captured the underlying patterns and trends in the sales data. This strong performance is further supported by the low RMSE, MAE, and MAPE values:

```python
Mean Absolute Error: 45127.23
RMSE: 64428.33
Mape: 4.56
```

## Improvements and Recommendations
- **Incorporate More Data**: The analysis was based on three years of weekly sales data, which provided a solid foundation for the model. However, incorporating a longer historical period could further enhance the model's ability to capture long-term trends and seasonal patterns, leading to even more accurate forecasts.

- **Add More Variables**: The current model only utilized holiday indicators and temperature as additional regressors. Including more relevant variables, such as promotional events, economic indicators, or regional factors, could provide the model with additional context, allowing it to better understand and predict sales fluctuations.

- **Optimize Hyperparameter Tuning**: While the model performed well with the chosen settings, further exploration of hyperparameters, such as adjusting the number of changepoints, changing the seasonality mode, or tweaking the prior scales, could help in fine-tuning the model.

## Contributors
Husani Newbold (Author)



