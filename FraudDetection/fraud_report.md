# Forensic Audit & Fraud Detection Report: Restaurant Menu Engineering & Profitability Analysis

## 1. Executive Summary

This forensic audit report details the methodology and findings of a comprehensive fraud and anomaly detection analysis conducted on the **Restaurant Menu Engineering and Profitability** master dataset (`MasterFoodBeverage_Data.xlsx`). The audit scanned **52,541 sales transactions** across five operating branches spanning a three-year period (2022–2024).

The investigation revealed systematic, high-impact anomalies that point to direct financial leakage and operational fraud:
1. **Revenue Skimming (POS Manipulation)**: **190 critical transactions** where bulk orders (quantities of 20–50 items) were registered, but the system calculated net sales, COGS, and profit based on single-digit quantities (2, 3, or 4 items). This deliberate mismatch between transaction quantity and total sales resulted in a verified net revenue leakage of **Rs. 2,371,533.04** and profit leakage of **Rs. 1,666,154.34** (representing **4.60% of the entire restaurant chain's reported profit**).
2. **Unauthorized Sweethearting (Employee Discounts)**: **158 suspicious transactions** featuring extremely high discounts of **60% to 90%** applied using irregular, non-standard percentages (e.g., 61%, 62%, 64%, etc.) that bypass standard 5% POS increments.
3. **Overcharging / Price Inflation**: **666 transactions** where customers were overcharged by more than 20% compared to the standard menu base price, exposing the restaurant chain to severe brand risk and audit non-compliance.

Statistically, Benford's Law analysis confirmed that **four out of five branches (Downtown, Airport, Beach, and Mall) exhibit non-conformity (High Risk of tampering)**, while only the Suburb branch conforms to natural transaction distributions.

---

## 2. Dataset & Data Cleaning

The audit ingested the multi-tab Excel sheet containing:
* `Sales_Fact`: 52,541 records representing transaction sales.
* `Customer_Dim`: 3,000 unique customer profiles.
* `Menu_Dim` & `Category_Dim`: 30 menu items denormalized into 6 food categories.
* `Location_Dim`: 5 operating branches (Downtown, Airport, Beach, Mall, Suburb).
* `Date_Dim`: 1,096 days of date mapping.

### Data Cleaning and Ingest Processing
Before applying audit logic, we ensured complete structural integrity:
1. **Whitespace Trimming**: All string values in fields like `branch_name`, `payment_method`, and `order_type` were stripped of leading/trailing spaces.
2. **Standardization of Key Fields**: Datetime fields were standardized. Dimensions were joined with the fact table via primary keys (`item_id`, `location_id`) to ensure base prices and branch names mapped correctly.
3. **No Missing Values**: The dataset was verified to have zero missing values in critical numeric fields.

---

## 3. Audit Methodology: Detection Rules & Findings

The investigation combined strict rule-based triggers with unsupervised machine learning to isolate suspect transactions.

### A. Rule-Based Flags

#### Rule 1: Revenue Skimming (POS Tampering)
* **Trigger Condition**: Transactions with `quantity >= 20` where the recorded `net_sales` deviates from `correct_net_sales` (calculated as `quantity * unit_price * (1 - discount_pct/100)`) by more than Rs. 0.05.
* **Mechanism**: In these 190 transactions, the cashier registered a bulk order (e.g., quantity of 46 items) but manipulated the final sales entry. The system calculated `net_sales` and `cogs` based on a small quantity (e.g., 4 items), leaving the actual `quantity` recorded as 46 but pocketing the cash value for the other 42 items.
* **Audit Finding**: **190 transactions** flagged. Total net sales leakage is **Rs. 2,371,533.04** and profit leakage is **Rs. 1,666,154.34**.

*Mathematical Example (Row index 95 in raw data)*:
* Recorded Quantity: **46** | Unit Price: **Rs. 345.90** | Discount %: **10%**
* Recorded Gross Sales: **Rs. 15,911.40** (Correctly calculated as $46 \times 345.90$)
* Recorded Net Sales: **Rs. 1,245.24** (Artificially calculated as $4 \times [345.90 \times (1 - 0.10)]$)
* Actual Revenue Should Be: **Rs. 14,320.26**
* **Revenue Leakage (Skimmed)**: **Rs. 13,075.02** for this single order.

#### Rule 2: Sweethearting (Unauthorized Discounts)
* **Trigger Condition**: Transactions with a `discount_pct >= 50%`.
* **Mechanism**: Employees giving away unauthorized free or heavily discounted food to friends/accomplices. Standard customer promotions are limited to 0%, 5%, 10%, 15%, 20%, 25%, and 30%.
* **Audit Finding**: **158 transactions** flagged, with discounts ranging from 60% to 90% (e.g., 61%, 62%, 64%, etc.), indicating a bypass of POS controls.

#### Rule 3: Unit Price Inflation (Overcharging)
* **Trigger Condition**: Transactions where actual `unit_price` in the sales fact is more than 20% higher than the menu `base_price` (Price Ratio > 1.20).
* **Mechanism**: Cashiers charging customers more than the standard menu price, likely pocketing the difference or artificially boosting sales margins.
* **Audit Finding**: **666 transactions** flagged, with a maximum inflation ratio of **1.2074** (20.74% overcharging) affecting popular items like *Pizza Margherita* (39 times) and *Butter Chicken* (38 times).

#### Rule 4: Large Order Outliers
* **Trigger Condition**: Transactions with `quantity >= 20` that are *not* flagged as skimming (i.e. calculated correctly).
* **Audit Finding**: Only **2 transactions** met this criteria. This confirms that almost all large quantity orders (190 out of 192) in the raw dataset were manipulated for revenue skimming.

---

### B. Statistical & ML Anomaly Detection

#### 1. Benford's Law (First-Digit Analysis)
Benford's Law states that in naturally occurring financial datasets, the number 1 appears as the first digit in about 30.1% of cases, while 9 appears in only 4.6% of cases. Forensic auditors use the **Mean Absolute Deviation (MAD)** to measure deviation:
$$MAD = \frac{1}{9} \sum_{d=1}^9 |P_{actual}(d) - P_{benford}(d)|$$

* **Audit Thresholds**: MAD < 0.012 is acceptable conformity; MAD >= 0.015 indicates statistical non-conformity (tampering/manipulation).
* **Findings**: 
  * **Suburb Branch**: MAD = **0.0096** $\rightarrow$ **Acceptable Conformity (Low Risk)**
  * **Airport Branch**: MAD = **0.0152** $\rightarrow$ **Non-Conformity (High Risk)**
  * **Mall Branch**: MAD = **0.0177** $\rightarrow$ **Non-Conformity (High Risk)**
  * **Beach Branch**: MAD = **0.0185** $\rightarrow$ **Non-Conformity (High Risk)**
  * **Downtown Branch**: MAD = **0.0186** $\rightarrow$ **Non-Conformity (High Risk)**
  * *Interpretation*: Four out of five branches have net sales distributions that deviate significantly from natural patterns, confirming widespread transaction tampering.

#### 2. Z-Score & IQR Outliers
* **Z-Score**: Flagged **1,251 transactions** with net sales values exceeding 3 standard deviations from the mean (Z-Score > 3), indicating extreme transaction values.
* **IQR (Interquartile Range)**: Flagged **2,423 transactions** outside the range $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$ (Upper limit of Rs. 2,786.35).

#### 3. Isolation Forest (Multivariate Machine Learning)
An unsupervised Isolation Forest model was trained on 7 features: `quantity`, `unit_price`, `discount_pct`, `net_sales`, `cogs`, `profit`, and `customer_rating`. 
* **Findings**: The model isolated the **1,051 most anomalous transactions** (2.0% contamination). The ML model successfully flagged the skimming transactions and identified multivariate anomalies, such as transactions with perfect customer ratings (5.0) that had abnormally high discount percentages or very low profits.

---

## 4. Financial & Risk Analysis by Branch

The table below summarizes the audited risk profile and financial leakage across all branches:

| Branch Name | Scanned Transactions | Flagged Skimming | Net Sales Leakage (Rs.) | Profit Leakage (Rs.) | Benford MAD | Risk Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Airport** | 10,500 | 43 | 600,357.41 | 420,853.18 | 0.0152 | **High Risk** |
| **Beach** | 10,480 | 42 | 590,460.52 | 427,328.03 | 0.0185 | **High Risk** |
| **Downtown** | 10,591 | 45 | 507,813.25 | 351,737.40 | 0.0186 | **High Risk** |
| **Mall** | 10,470 | 36 | 419,456.77 | 298,999.34 | 0.0177 | **High Risk** |
| **Suburb** | 10,500 | 24 | 253,445.09 | 167,236.39 | 0.0096 | **Low Risk** |
| **TOTAL** | **52,541** | **190** | **2,371,533.04** | **1,666,154.34** | **-** | **-** |

### Leakage by Payment Method
Auditing the payment channels revealed that skimming occurs across all payment methods, indicating it is a POS system/software loophole rather than cashier cash skimming alone:
* **UPI**: 41 skimming transactions | **Rs. 561,763.70** leakage
* **Online**: 38 skimming transactions | **Rs. 486,056.61** leakage
* **Cash**: 39 skimming transactions | **Rs. 457,343.66** leakage
* **Card**: 37 skimming transactions | **Rs. 457,180.07** leakage
* **Wallet**: 35 skimming transactions | **Rs. 409,189.01** leakage

### Top Items Exploited for Skimming
High-value menu items were heavily targeted for skimming, resulting in disproportionate losses:
1. **Family Feast**: 5 skimming transactions | **Rs. 206,283.41** leakage
2. **Couple Dinner**: 7 skimming transactions | **Rs. 195,503.11** leakage
3. **Pizza Margherita**: 14 skimming transactions | **Rs. 162,303.66** leakage
4. **Butter Chicken**: 11 skimming transactions | **Rs. 161,731.81** leakage
5. **Truffle Mushroom Risotto**: 7 skimming transactions | **Rs. 147,257.20** leakage

---

## 5. Audit Recommendations & Mitigation Plan

To plug these financial leaks and prevent future occurrences, the following systemic controls must be implemented immediately:

1. **POS Mathematical Validation Locks**: 
   * **Issue**: The POS system currently allows writing a `net_sales` value that is mathematically inconsistent with the `quantity`, `unit_price`, and `discount_pct`.
   * **Control**: Modify the POS database schema and software to calculate `net_sales` and `cogs` via hardcoded database triggers or application constraints:
     $$\text{net\_sales} = \text{quantity} \times \text{unit\_price} \times \left(1 - \frac{\text{discount\_pct}}{100}\right)$$
     This field must be read-only and never editable by cashiers or POS terminals.
2. **Discount Authorization & Promo Code Controls**:
   * **Issue**: Custom, non-standard discounts (e.g. 61%, 86%) are applied at the terminal, representing "sweethearting".
   * **Control**: Hardcode a dropdown menu of allowed discount rates (0%, 5%, 10%, 15%, 20%, 25%, 30%). Any custom discount rate or discount >= 50% must require a manager's physical RFID card scan or supervisor authentication code.
3. **Menu Price Lock & Syncing**:
   * **Issue**: Cashiers are inflating unit prices by up to 20% (e.g. charging Rs. 265 for Masala Chai instead of Rs. 220).
   * **Control**: Lock the `unit_price` field in the POS. Unit prices must automatically inherit from the centralized master `Menu_Dim` table and cannot be overridden manually at the register.
4. **Automated Daily Audit Reporting**:
   * **Control**: Set up a cron job or scheduled script (similar to `run_fraud_detection.py`) to run nightly. If any transaction contains a mathematical inconsistency or a price ratio > 1.0, the transaction must be flagged, and an automated alert email must be sent directly to internal audit.
5. **Customer Receipt Verification**:
   * **Control**: Encourage customers to check their printed receipts. Ensure the receipt displays the exact quantity and price, and print a QR code linking to a digital receipt generator to prevent cashiers from printing "draft" receipts and editing the order afterward.
