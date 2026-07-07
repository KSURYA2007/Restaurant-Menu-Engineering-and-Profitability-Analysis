# Technical Methodology & Insights Report: Key KPI Command Center Dashboard

## 1. Executive Summary

This technical report details the design methodology, financial logic, and strategic insights behind the **Key KPI Command Center Dashboard** created for the **Restaurant Menu Engineering and Profitability Analysis** project. 

The dashboard serves as a high-fidelity operational and strategic "Command Center" monitoring critical business indicators across **52,541 sales transactions** spanning a three-year period (2022–2024).

The restaurant chain demonstrates healthy overall performance:
* **Total Net Sales**: **Rs. 54,551,551.95** (Rs. 54.55M)
* **Total Operating Profit**: **Rs. 38,091,265.20** (Rs. 38.09M)
* **Overall Profit Margin**: **69.83%**
* **Total Unique Orders**: **15,000**
* **Average Order Value (AOV)**: **Rs. 3,636.77**
* **Average Customer Rating**: **4.09 / 5.0 stars**

Forensic analysis highlights **Beach** as the most efficient branch (highest revenue and profit per seat), and **Main Course** as the largest strategic revenue driver, contributing **Rs. 19.50 Million** to net sales.

---

## 2. Dashboard Dimension Mapping & KPIs

To provide a holistic view of the business, the Command Center aggregates metrics across two core dimensions:

### A. Strategic Dimensions (Financial & Menu Health)
These metrics measure the overall health, profitability, and growth trajectories of the business:
1. **Net Revenue (Net Sales)**: Total income generated after discount deductions.
   $$\text{Net Sales} = (\text{Quantity} \times \text{Unit Price}) - \text{Discount Amount}$$
2. **Operating Profit**: Net revenue remaining after subtracting Cost of Goods Sold (COGS).
   $$\text{Operating Profit} = \text{Net Sales} - (\text{Quantity} \times \text{Unit Cost})$$
3. **Operating Profit Margin (%)**: The efficiency of converting sales into profit.
   $$\text{Profit Margin} = \left( \frac{\text{Operating Profit}}{\text{Net Sales}} \right) \times 100$$
4. **Month-over-Month (MoM) Growth Rate**: Percentage change in monthly sales/profit.
5. **BCG Matrix Classification**: Evaluation of menu items based on relative popularity (quantity) and unit profitability.

### B. Operational Dimensions (Efficiency & Channels)
These metrics monitor day-to-day fulfillment, channel mix, and resource utilization:
1. **Average Order Value (AOV)**: Average net spending per order.
   $$\text{AOV} = \frac{\text{Total Net Sales}}{\text{Total Unique Orders}}$$
2. **Branch Seating Capacity Utilization**: The financial productivity of physical seating space.
   $$\text{Revenue per Seat} = \frac{\text{Branch Revenue}}{\text{Branch Seating Capacity}}$$
   $$\text{Profit per Seat} = \frac{\text{Branch Operating Profit}}{\text{Branch Seating Capacity}}$$
3. **Channel Splits (Order Type & Payment Method)**: Market share analysis of order channels (Dine-In, Takeaway, Delivery) and transaction options (Cash, Cards, UPI, Online, Wallets).
4. **Weekly Order Patterns**: Identification of peak traffic days (Monday–Sunday).

---

## 3. Menu Engineering: BCG Matrix Analysis

Menu items were categorized into four quadrants using the Growth-Share BCG matrix logic based on two benchmarks:
* **Popularity Benchmark (Quantity Sold)**: **4,571.70 units** (Average sales volume per menu item).
* **Profitability Benchmark (Unit Margin)**: **Rs. 307.86** (Average unit contribution margin per item).

$$\text{Unit Contribution Margin} = \frac{\text{Total Net Sales} - \text{Total COGS}}{\text{Total Quantity Sold}}$$

### BCG Quadrant Definitions & Item Distribution:

1. **Stars (High Popularity, High Margin)**:
   * *Strategic Role*: Core profit drivers. Maintain high quality and promote actively.
   * *Count*: 5 menu items.
   * *Top Items*: *Lamb Biryani*, *Chicken Tikka Masala*, *Family Feast*, *Veg Combo Meal*, *Couple Dinner*.
2. **Plowhorses (High Popularity, Low Margin)**:
   * *Strategic Role*: Revenue volume drivers. Consider slight price increases (+5%) or portion adjustments to improve margin without hurting popularity.
   * *Count*: 10 menu items.
   * *Top Items*: *Veg Spring Rolls*, *Butter Chicken*, *Dal Makhani*, *Veg Biryani*, *Pizza Margherita*.
3. **Puzzles (Low Popularity, High Margin)**:
   * *Strategic Role*: High-margin opportunities. Leverage staff recommendations, placement on menu design, or promotional discounts to boost quantity sold.
   * *Count*: 5 menu items.
   * *Top Items*: *Grilled Salmon*, *Chef's Special Curry*, *Truffle Mushroom Risotto*, *Seared Duck Breast*, *Lobster Thermidor*.
4. **Dogs (Low Popularity, Low Margin)**:
   * *Strategic Role*: Portfolio drag. De-emphasize on menu, re-engineer recipe to reduce cost, or replace entirely.
   * *Count*: 10 menu items.
   * *Top Items*: *Paneer Tikka*, *Chicken Wings*, *Fish Tacos*, *Soup of the Day*, *Pasta Arrabiata*.

---

## 4. Operational & Strategic Insights

### A. Branch Efficiency Rankings
Analyzing branch capacity reveals that larger branches are not necessarily the most efficient:

* **Beach Branch (60 seats)**: Generated **Rs. 11.79M** (highest revenue) and **Rs. 8.50M** profit. It achieved a staggering **Rs. 196,456.64 Revenue per Seat** and **Rs. 141,652.25 Profit per Seat**, making it the chain's star performer.
* **Airport Branch (80 seats)**: Generated **Rs. 11.37M** revenue, achieving a high **Rs. 142,153.76 Revenue per Seat**.
* **Suburb Branch (90 seats)**: Performed moderately, with **Rs. 9.54M** revenue and **Rs. 105,998.58 Revenue per Seat**.
* **Downtown Branch (120 seats)**: Performed below average relative to capacity, yielding **Rs. 90,718.23 Revenue per Seat**.
* **Mall Branch (200 seats)**: Performed poorest in efficiency, generating **Rs. 10.97M** revenue but yielding only **Rs. 54,828.96 Revenue per Seat** due to overcapacity of seating.

### B. Category Contribution
* **Main Courses** and **Combos** dominate sales, contributing **63.15%** of total net sales (Rs. 19.50M and Rs. 14.95M respectively).
* **Beverages** and **Desserts** represent high-margin niches (**78.69%** and **74.72%** respectively) and should be aggressively upsold.

### C. Customer Intelligence
* **Loyal Customers** and **Regular Customers** represent the backbone of profitability, generating **78.2%** of total operating profit.
* Scatter plot analysis confirms a strong linear correlation between a customer's **Loyalty Points** and their **Lifetime Net Spending (Rs.)**, validating the efficiency of the restaurant's customer loyalty program.

---

## 5. Strategic Recommendations

1. **Downsize / Rationalize Seating in Mall Branch**:
   * The Mall branch has 200 seats but only matches the revenue of the 60-seat Beach branch. Reducing active seating in Mall by 40% will save labor and maintenance costs without impacting peak revenue.
2. **Promote Puzzles via Staff Training**:
   * Items like *Lobster Thermidor* and *Truffle Mushroom Risotto* have high contribution margins but low sales volume. Incentivize staff to suggest these items to customers.
3. **Re-price/Modify Plowhorses**:
   * *Butter Chicken* and *Dal Makhani* are extremely popular but have low margins. Consider a modest 5% price increase, which would significantly improve overall profitability due to high volumes.
4. **Target upsells of Beverages and Desserts**:
   * Beverages have the highest profit margin (78.69%). Training staff to suggest beverages or desserts to dine-in customers will instantly increase AOV (currently Rs. 3,636.77) and overall chain margin.
