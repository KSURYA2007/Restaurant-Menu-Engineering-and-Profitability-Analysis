# Sales Forecasting using Time Series Analysis - Technical Report

## 1. Executive Summary

This report details the methodology, technical design, and business findings of the **90-Day Sales Forecasting** project for the **Restaurant Menu Engineering and Profitability Analysis** chain.

The forecasting model aggregates three years of daily transactions (**1,096 consecutive days** from 2022-01-01 to 2024-12-31) and projects sales for Q1 2025 (90 days, 2025-01-01 to 2025-03-31) with 95% confidence intervals.

### Key Forecast Highlights:
* **Total Forecasted Q1 2025 Sales**: **Rs. 4,374,213.68**
* **Average Daily Sales Projection**: **Rs. 48,602.37**
* **Daily Forecast Error Standard Deviation (RMSE)**: **Rs. 17,263.14**
* **Projected Sales Peak**: **Rs. 56,128.75** (on Tuesday, 2025-02-11)
* **Projected Sales Trough**: **Rs. 40,782.95** (on Friday, 2025-01-10)

---

## 2. Time Series Properties & Stationarity Analysis

Before choosing the forecasting model, we conducted a rigorous statistical examination of the daily Net Sales series ($Y_t$):

### A. Autocorrelation & Stationarity
We evaluated the Pearson correlation coefficient between $Y_t$ and its historical lags:
* **Lag 1 Autocorrelation**: $-0.0355$
* **Lag 2 Autocorrelation**: $-0.0228$
* **Lag 7 Autocorrelation**: $+0.0637$
* **Lag 30 Autocorrelation**: $+0.0073$

The extremely low lag correlations (all close to zero) indicate that after accounting for minimal calendar effects, the daily sales behave as a **stationary, random-walk / white noise series** around a constant mean. There is no strong momentum, autoregressive dependency, or lag memory. This rules out complex autoregressive models (like ARIMA with high $p$ or $q$) which would overfit the noise and quickly decay to the mean.

### B. Trend Analysis
Evaluating the average daily sales by year shows a slight downward trajectory:
* **2022 Average Daily Sales**: Rs. 51,043.45 (Total: Rs. 18.63 Million)
* **2023 Average Daily Sales**: Rs. 49,295.40 (Total: Rs. 17.99 Million)
* **2024 Average Daily Sales**: Rs. 48,983.26 (Total: Rs. 17.93 Million)

There was a $-3.42\%$ decline from 2022 to 2023, followed by a minor $-0.63\%$ decline from 2023 to 2024. This linear trend is modeled using a deterministic time trend feature.

---

## 3. Mathematical Methodology

We used a **Direct Ridge Regression Forecasting Model** combining the deterministic trend and calendar seasonal dummy features:

### A. Model Equation
For day $t$, the forecasted daily sales $\hat{y}_t$ is modeled as:
$$\hat{y}_t = \beta_0 + \beta_1 \cdot \text{trend}_t + \sum_{d=0}^{6} \alpha_d \cdot D_{d,t} + \sum_{m=1}^{12} \gamma_m \cdot M_{m,t}$$
Where:
* $\text{trend}_t$ is the day index ($0, 1, \dots, 1095$ for training; $1096, \dots, 1185$ for the Q1 2025 forecast).
* $D_{d,t}$ is a dummy variable indicating the day of the week $d \in \{0, \dots, 6\}$.
* $M_{m,t}$ is a dummy variable indicating the month of the year $m \in \{1, \dots, 12\}$.
* Ridge regularization (L2 penalty $\lambda = 5.0$) was applied to stabilize the coefficient estimates and prevent multi-collinearity.

### B. Daily Confidence Intervals
Because the daily sales are highly stationary and the residuals are independent and identically distributed, the forecast standard error for any future day $t$ is constant and equal to the model's training Root Mean Squared Error (RMSE):
$$\sigma_t = \text{RMSE} = \text{Rs. 17,263.14}$$
The 95% Confidence Interval for each daily prediction is:
$$\text{CI}_{daily,t} = \hat{y}_t \pm 1.96 \times \text{RMSE} = \hat{y}_t \pm \text{Rs. 33,835.75}$$

### C. Scaling for Aggregated Forecasts (Weekly & Monthly)
When daily forecasts are summed to compile weekly ($N=7$) or monthly ($N \in \{28, 30, 31\}$) totals, the standard error scales based on the independence of daily errors:
$$\text{Var}\left(\sum_{t=1}^N e_t\right) = N \times \text{Var}(e) = N \times \text{RMSE}^2$$
$$\text{SE}_{agg} = \text{RMSE} \times \sqrt{N}$$
Therefore, the 95% Confidence Intervals for weekly and monthly sums are:
* **Weekly Sum Confidence Interval**:
  $$\hat{Y}_{weekly} \pm 1.96 \times \text{RMSE} \times \sqrt{7} = \hat{Y}_{weekly} \pm \text{Rs. 89,520.10}$$
* **Monthly Sum Confidence Interval**:
  $$\hat{Y}_{monthly} \pm 1.96 \times \text{RMSE} \times \sqrt{N}$$
  For January ($N=31$): $\hat{Y}_{Jan} \pm \text{Rs. 188,367.65}$
  For February ($N=28$): $\hat{Y}_{Feb} \pm \text{Rs. 179,042.86}$

This scaling mathematically accounts for the averaging of random noise over time, leading to much tighter relative confidence intervals at aggregated levels.

---

## 4. Key Findings & Diagnostic Implications

### A. Weekday Seasonality Coefficients (Deviation from Mean)
* **Tuesdays**: **+Rs. 2,859.50** (Positive peak)
* **Thursdays**: **+Rs. 2,328.50**
* **Sundays**: **+Rs. 1,876.45**
* **Saturdays**: **-Rs. 3,171.81** (Negative peak)

*Operational Insight*: Surprisingly, Saturdays represent the lowest average sales day for the chain (contrary to typical restaurant trends, which might be due to location-specific behaviors, such as business districts closing on weekends), whereas Tuesdays and Thursdays represent the peak activity. Staffing levels should be adjusted accordingly.

### B. Monthly Seasonality Coefficients
* **August**: **+Rs. 1,875.97** (Summer peak)
* **October**: **-Rs. 3,758.71** (Fall trough)
* **May**: **-Rs. 1,612.64**

---

## 5. Strategic Recommendations

1. **Labor Scheduling Optimization**:
   Schedule extra servers and kitchen staff on Tuesdays and Thursdays. Reduce staffing levels on Saturdays and Mondays to cut down on operational labor costs.
2. **October Margin Buffer**:
   Since October is historically a major seasonal trough, launch autumn-themed promotional events, corporate group bookings, or seasonal discounts to offset the expected Rs. 3,700 daily sales drop.
3. **Inventory Management**:
   Procurement should align with the weekly forecasting models rather than daily trends to avoid overstocking, as daily noise is high, but weekly totals are highly predictable with a standard error of only Rs. 45,673 (about 12% of average weekly sales).
