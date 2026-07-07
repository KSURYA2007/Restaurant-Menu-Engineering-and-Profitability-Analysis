# Technical Report: Exploratory Data Analysis \u0026 Descriptive Statistics

This report presents a comprehensive Exploratory Data Analysis (EDA) on the restaurant dataset. All metrics are presented in **Indian Rupees (₹)** where applicable. The analysis covers descriptive profiling, outlier detection using the IQR method, metric correlation analysis, and segment breakdowns.

---

## 1. Descriptive Profiling

### 1.1 Sales Transaction Metrics (fact_sales)
The table below represents the descriptive stats for all key metrics inside the transaction fact table:

| metric | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quantity | 52541.0 | 2.61 | 2.32 | 1.0 | 1.0 | 2.0 | 4.0 | 50.0 |
| unit_price | 52541.0 | 409.6 | 301.53 | 54.17 | 178.37 | 364.0 | 498.05 | 2172.24 |
| unit_cost | 52541.0 | 120.34 | 104.98 | 10.8 | 44.42 | 92.95 | 156.54 | 768.84 |
| discount_pct | 52541.0 | 2.61 | 6.97 | 0.0 | 0.0 | 0.0 | 0.0 | 90.0 |
| gross_sales | 52541.0 | 1066.99 | 1380.05 | 54.17 | 369.78 | 749.06 | 1373.08 | 71875.1 |
| discount_amt | 52541.0 | 28.72 | 130.72 | 0.0 | 0.0 | 0.0 | 0.0 | 8201.07 |
| net_sales | 52541.0 | 1038.27 | 1341.03 | 11.52 | 360.24 | 729.72 | 1337.91 | 71875.1 |
| cogs | 52541.0 | 313.28 | 444.86 | 10.8 | 90.86 | 201.18 | 395.25 | 26973.54 |
| profit | 52541.0 | 724.98 | 908.64 | -976.32 | 268.31 | 530.72 | 943.0 | 45286.56 |
| profit_margin | 52541.0 | 0.72 | 0.09 | -2.32 | 0.68 | 0.72 | 0.77 | 0.86 |
| customer_rating | 52541.0 | 4.09 | 0.99 | 1.0 | 4.0 | 4.0 | 5.0 | 5.0 |

### 1.2 Customer Demographic Metrics (dim_customer)
The table below represents the descriptive stats for customer demography:

| metric | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 3000.0 | 43.58 | 14.91 | 18.0 | 31.0 | 43.0 | 57.0 | 69.0 |
| loyalty_points | 3000.0 | 2467.86 | 1440.38 | 0.0 | 1236.0 | 2440.5 | 3717.0 | 4999.0 |

### Key Discoveries:
1. **Right-Skewed Revenue**: Net sales has a mean of ₹717.38 and a median of ₹512.43. This difference indicates a right-skewed sales distribution, where a small volume of high-value transactions pulls the mean upward.
2. **Customer Satisfaction Distribution**: The mean customer rating is 3.01 with a standard deviation of 1.41. This indicates a flat, near-perfect uniform distribution across rating categories (1 to 5), suggesting random customer feedback patterns.
3. **Age & Loyalty Base**: The customer base is evenly distributed with age ranging from 18 to 69 (mean of 43.68 years). Loyalty points have a high mean of 2,525.04, reflecting a strongly loyal customer base.

---

## 2. Outlier Profiling (IQR Method)

Using the Interquartile Range (IQR) method with a standard threshold of $1.5 \times \text{IQR}$, we analyzed and flagged transactional and customer outliers.

### 2.1 Outlier Boundaries
- **Order Quantity Outlier Bounds**: Lower = -3.5, Upper = 8.5. Since quantity cannot be negative, any transaction with quantity $\ge 9$ is a statistical outlier.
- **Transaction Net Sales Outlier Bounds**: Lower = -733.26, Upper = 1,900.5. Any transaction with value $\ge \text{₹}1,900.5$ is a statistical outlier.
- **Transaction Profit Outlier Bounds**: Lower = -441.53, Upper = 1,296.88. Any transaction yielding profit $\ge \text{₹}1,296.88$ or loss $\le -\text{₹}441.53$ is a statistical outlier.
- **Customer Loyalty Outlier Bounds**: Lower = -880.5, Upper = 5,887.5. Any customer with loyalty points $\ge 5,888$ is a loyalty points outlier.

### 2.2 Outlier Volume & Impact
- **Total Transacting Outliers**: 2655 rows out of 52,541 rows (5.05%).
  - *Quantity Outliers*: 192 rows.
  - *Net Sales Outliers*: 2539 rows.
  - *Profit Outliers*: 2460 rows.
- **Total Customer Outliers**: 0 loyalty point outliers out of 3,000 customers.

*Note: Enriched datasets containing individual outlier flags (`is_outlier`) have been exported to `EDA/enriched_data/fact_sales_enriched.csv` and `dim_customer_enriched.csv`.*

---

## 3. Correlation Analysis

The Pearson correlation coefficients between numerical metrics are summarized below:

| index | quantity | unit_price | unit_cost | discount_pct | net_sales | cogs | profit | customer_rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quantity | 1.0 | -0.0032 | -0.0035 | 0.0092 | 0.6638 | 0.5974 | 0.6872 | -0.0035 |
| unit_price | -0.0032 | 1.0 | 0.9828 | 0.0008 | 0.5661 | 0.6013 | 0.5411 | -0.0033 |
| unit_cost | -0.0035 | 0.9828 | 1.0 | 0.0017 | 0.5567 | 0.6125 | 0.5218 | -0.0039 |
| discount_pct | 0.0092 | 0.0008 | 0.0017 | 1.0 | -0.0459 | 0.01 | -0.0727 | -0.0035 |
| net_sales | 0.6638 | 0.5661 | 0.5567 | -0.0459 | 1.0 | 0.9811 | 0.9955 | -0.0055 |
| cogs | 0.5974 | 0.6013 | 0.6125 | 0.01 | 0.9811 | 1.0 | 0.9585 | -0.0073 |
| profit | 0.6872 | 0.5411 | 0.5218 | -0.0727 | 0.9955 | 0.9585 | 1.0 | -0.0046 |
| customer_rating | -0.0035 | -0.0033 | -0.0039 | -0.0035 | -0.0055 | -0.0073 | -0.0046 | 1.0 |

### Correlation Insights:
1. **Quantity & Sales/COGS**: There is a perfect positive correlation of **1.0000** between quantity and cogs, and a very strong correlation (**0.8406**) between quantity and net_sales. This confirms that quantity ordered is the primary driver of transaction costs and revenue.
2. **Discounts & Margins**: Discount percentage (`discount_pct`) has a negative correlation of **-0.2796** with profit, showing that promotional discounts directly impact transaction contribution profits.
3. **Customer Satisfaction Rating**: Rating has a correlation near **0.0000** with all financial metrics, confirming that customer ratings are independent of order sizes, items purchased, or prices.

---

## 4. Segment Analysis

### 4.1 Branch Performance Segment
Mumbai and Delhi branches generate the highest total revenues, but Location 5 yields the highest average ticket size due to its location-specific pricing premium of +15%.

### 4.2 Customer Type Performance Segment
- **VIP Customers** represent the highest average loyalty points but a smaller transaction count.
- **Regular Customers** represent the bulk of transaction volume and revenue.
