# Forensic & Strategic Command Center: Grand Unified Interactive Dashboard Report

## 1. Executive Summary

This report details the methodology, technical design, and business findings of the **Grand Unified Interactive Dashboard** developed for the **Restaurant Menu Engineering and Profitability Analysis** project. 

The primary goal of this dashboard is to integrate all previous analyses—Exploratory Data Analysis (EDA), Extract-Transform-Load (ETL) pipelines, RFM customer behavioral segmentation, product performance/price elasticity, and forensic fraud auditing—into a single, cohesive dashboard that tells a complete business story.

The dashboard integrates **52,541 transactional records** and **3,000 customer profiles** into a single, interactive hub. The key high-level findings include:
* **Audited Net Revenue**: **Rs. 54,551,551.95** (Rs. 54.55M)
* **Audited Operating Profit**: **Rs. 38,091,265.20** (Rs. 38.09M)
* **Chain-wide Operating Profit Margin**: **69.83%**
* **Total Transactions**: **15,000 unique orders**
* **Average Customer Rating**: **4.09 / 5.0 stars**
* **Verified Financial Leakage**: **Rs. 2,371,533.04** net sales (and **Rs. 1,666,154.34** profit) underreported due to revenue skimming.

---

## 2. Integration Mapping & Data Schema

The unified database connects the transaction log (`Sales_Fact`) with customer profiles, menu details, branches, and date properties. The pipeline integrates 6 analytical layers:

```
┌────────────────────────────────────────────────────────┐
│                      SALES_FACT                        │
│  Recalculated gross/net sales, cogs, profit, margins   │
└───────────┬──────────────┬──────────────┬──────────────┘
            │              │              │
 ┌──────────▼───┐ ┌────────▼─────┐ ┌──────▼───────┐
 │   MENU DIM   │ │ CUSTOMER DIM │ │ FORENSIC AUDIT│
 │ BCG Segment  │ │ RFM Segment  │ │ Skimming/    │
 │ (Star, Dog,  │ │ (Champions,  │ │ Inflation    │
 │ Plowhorse)   │ │ At Risk, etc)│ │ ML Outliers  │
 └──────────────┘ └──────────────┘ └──────────────┘
```

1. **Financial Consistency Layer (ETL)**: Math correction formulas for gross sales, net sales, COGS, and profit round all figures to 2 decimal places.
2. **Menu Engineering Layer (BCG Matrix)**: Menu items categorized into Stars, Plowhorses, Puzzles, and Dogs based on quantity sold and unit contribution margins.
3. **Behavioral Customer Layer (RFM Segmentation)**: Customers grouped into 10 behavioral profiles based on Recency, Frequency, and Monetary quintile scores.
4. **Forensic Audit Layer (Rule-Based Flags)**: Transactions flagged for skimming, sweethearting, and price inflation.
5. **Statistical Verification Layer (Benford's Law)**: Evaluation of digit distributions to quantify tampering risk.
6. **Machine Learning Layer (Isolation Forest)**: Multi-dimensional outlier scoring to isolate complex, suspicious transaction combinations.

---

## 3. Methodologies

### A. RFM Quintiles (Customer Segmentation)
Customer transaction history was aggregated to calculate:
* **Recency (R)**: Days elapsed between the final transaction date in the dataset (2024-12-31) and the customer's last purchase.
* **Frequency (F)**: Unique orders placed by the customer.
* **Monetary (M)**: Total audited net spending by the customer.

Customers were scored from 1 to 5 on each metric using quintile binning (`pd.qcut`). An RFM group code (e.g. `555`, `111`) was assigned. The scores were mapped to 10 customer behavior segments (e.g., *Champions* with high recency and frequency, *At Risk* with low recency but high historical frequency, *Lost* with low scores across all dimensions).

### B. BCG Matrix (Menu Analysis)
Popularity and unit contribution benchmarks are computed globally:
* **Popularity Benchmark**: Average sales quantity sold per item (**4,571.70 units**).
* **Profitability Benchmark**: Average unit contribution margin (**Rs. 307.86**).

Menu items are mapped:
* **Stars** (High Popularity, High Margin): Core menu drivers.
* **Plowhorses** (High Popularity, Low Margin): Volume drivers.
* **Puzzles** (Low Popularity, High Margin): Margin opportunities.
* **Dogs** (Low Popularity, Low Margin): Portfolio drag.

### C. Forensic Audit & ML Outliers
* **Skimming Check**: Flagged transactions where quantity >= 20 but recorded net sales matches a single-digit multiplier (2, 3, or 4 items).
* **Sweethearting Check**: Flagged transactions with discount >= 50%.
* **Inflation Check**: Flagged transactions where actual unit price / menu base price > 1.20.
* **Benford's Law MAD**: Mean Absolute Deviation of the first digit of net sales at each branch compared to Benford's distribution.
* **Isolation Forest**: Multi-feature anomaly classifier set to 2% contamination.

---

## 4. Key Strategic Insights & Findings

### A. Branch Efficiency Rankings
Forensic seating utilization shows that compact, high-efficiency branches outperform massive venues:
* **Beach (60 seats)**: Rs. 11.79M Net Sales | **Rs. 196,456.64 Revenue per Seat** | **Rs. 141,652.25 Profit per Seat**.
* **Airport (80 seats)**: Rs. 11.37M Net Sales | **Rs. 142,153.76 Revenue per Seat** | **Rs. 100,534.22 Profit per Seat**.
* **Mall (200 seats)**: Rs. 10.97M Net Sales | **Rs. 54,828.96 Revenue per Seat** | **Rs. 38,463.75 Profit per Seat**.
* *Insight*: The Mall branch suffers from severe overcapacity. Space should be downsized or leased out, as its revenue is matched by Beach branch with 70% fewer seats.

### B. Menu Diagnostics
* **Stars** (e.g. *Lamb Biryani*, *Chicken Tikka Masala*, and *Veg Combo Meal*) generate high sales and high margins. Keep recipes consistent and promote these items.
* **Plowhorses** (e.g. *Butter Chicken* and *Dal Makhani*) are popular but operate on thin margins. Consider a modest 5% price increase.
* **Puzzles** (e.g. *Truffle Mushroom Risotto* and *Lobster Thermidor*) have high margins but low popularity. Upsell these items through menu placement or staff incentives.

### C. Customer Demographics & Behavior
* **Loyal & Champions** segments represent only **28.4% of the customer base** but generate **58.2% of overalloperating profit**, demonstrating the importance of customer loyalty.
* Spending is heavily concentrated in the **25–34** and **35–44** age demographics.

### D. Fraud Audit
* Revenue skimming resulted in a verified leakage of **Rs. 2,371,533.04** in net sales.
* Benford's Law MAD indicates statistical non-conformity (**High Risk**) in four branches: Downtown ($MAD = 0.0186$), Beach ($MAD = 0.0185$), Mall ($MAD = 0.0177$), and Airport ($MAD = 0.0152$). Only Suburb shows acceptable conformity ($MAD = 0.0096$).
* The UPI channel accounted for the highest skimming losses (**Rs. 561,763.70**), followed by Online payments (**Rs. 486,056.61**).

---

## 5. Systemic Recommendations

1. **Implement Database Constraints**:
   POS software must enforce a strict, read-only calculation trigger:
   $$\text{net\_sales} = \text{quantity} \times \text{unit\_price} \times \left(1 - \frac{\text{discount\_pct}}{100}\right)$$
   This prevents cashiers from manual overrides in the sales table.
2. **Lock Unit Prices**:
   POS unit prices must automatically inherit from `dim_menu` and cannot be modified at checkout to eliminate overcharging.
3. **Manager Approvals for Discounts**:
   Discounts >= 50% or non-standard rates must require a manager's RFID swipe or override code.
4. **Targeted Customer Loyalty Outreach**:
   Leverage the RFM segments to execute automated email campaigns (e.g., win-back offers for *At Risk* spenders, loyalty multipliers for *Champions*).
