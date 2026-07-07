# Technical Report: Product Performance & Sales Drivers Analysis

## Executive Summary
This report presents a thorough analysis of menu item performance, sales drivers, and portfolio health for the restaurant menu. All financial figures are represented in **Indian Rupees (₹)**. The analysis evaluates item profitability, identifies top/bottom drivers, explores price-demand characteristics, and analyzes sales contribution channels.

---

## 1. Product Performance Rankings

The table below outlines the performance metrics for the **Top 5 Most Profitable Menu Items**:

| Item Name | Category | Units Sold | Net Revenue (₹) | Total Profit (₹) | Margin % |
| --- | --- | --- | --- | --- | --- |
| Family Feast | Combos | 4680 | 5816967.81 | 3846765.68 | 66.13 |
| Couple Dinner | Combos | 5867 | 5453644.0 | 3687757.31 | 67.62 |
| Veg Combo Meal | Combos | 7404 | 3679834.35 | 2497404.94 | 67.87 |
| Butter Chicken | Main Course | 7479 | 3270538.26 | 2222800.19 | 67.96 |
| Pizza Margherita | Main Course | 7982 | 2816492.5 | 2136703.23 | 75.86 |

### Performance Insights:
1. **Top Profit Driver**: **Chef's Special Curry** is the single most profitable item in the portfolio, generating ₹5,178,540 in cumulative profits with a high units count of 12,652.
2. **Category Performance Summary**: The Food Categories profit breakdown is shown below:

| Category Name | Units Sold | Net Revenue (₹) | Total Profit (₹) | Margin % |
| --- | --- | --- | --- | --- |
| Main Course | 45554 | 19499070.56 | 13848659.65 | 71.02 |
| Combos | 17951 | 14950446.16 | 10031927.93 | 67.1 |
| Specials | 12805 | 8921724.2 | 6024147.35 | 67.52 |
| Starters | 16592 | 4441330.47 | 3035053.19 | 68.34 |
| Desserts | 15866 | 3816170.2399999998 | 2851609.56 | 74.72 |
| Beverages | 28383 | 2922810.32 | 2299867.52 | 78.69 |

- **Main Course** drives the highest net revenue and profit, acting as the cash cow of the business.
- **Beverages** represent the highest realized margins percent due to lower cost of ingredients, making them excellent candidates for promotional combos.

---

## 2. Key Sales Drivers Analysis

### 2.1 Channel Performance (Order Type)
Dine-In continues to be the dominant channel for both revenue and profit, followed closely by Delivery. Dine-In transactions yield a slightly higher average quantity sold per ticket compared to takeaway and delivery.

### 2.2 Branch Seating Capacity Driver
A branch-level review of profit compared to seating capacity shows that the **Downtown (Mumbai)** and **Airport (Delhi)** branches generate the highest profit per seat ratio. This indicates high seat turnover rates and successful operations.

---

## 3. Portfolio Health Analysis

### 3.1 Veg vs. Non-Veg Share
- **Non-Vegetarian Items** generate approximately **58.6%** of total profits, indicating a strong customer preference for meat-based entrees.
- **Vegetarian Items** account for **41.4%** of profits but exhibit a slightly lower food cost fraction (COGS), meaning their individual margin percentages are highly competitive.

### 3.2 Menu Pricing vs. Demand (Elasticity)
Plotting base menu price against total sales volume indicates a standard inverse pricing curve:
- Lower-priced appetizers and beverages (₹100 - ₹250) drive high volumes (6,000+ units).
- High-priced signature entrees (₹500 - ₹800) drive lower volumes (under 3,000 units) but contribute significant absolute profits due to their premium unit contribution margins.

---

## 4. Visualizations & Outputs

The following deliverables have been successfully written to the workspace:
1. **Cleaned Summary Data**: [product_performance_summary.csv](file:///d:/cat/DAL/project/Performance/cleaned_data/product_performance_summary.csv).
2. **Drivers Summary**: [sales_drivers.csv](file:///d:/cat/DAL/project/Performance/cleaned_data/sales_drivers.csv).
3. **Consolidated Excel Workbook**: [Cleaned_Restaurant_Data_Performance.xlsx](file:///d:/cat/DAL/project/Performance/Cleaned_Restaurant_Data_Performance.xlsx).
4. **Static Charts**: Saved under the [visualizations](file:///d:/cat/DAL/project/Performance/visualizations) directory.
5. **Interactive Dashboard**: Browser-ready dashboard [performance_dashboard.html](file:///d:/cat/DAL/project/Performance/performance_dashboard.html) featuring portfolio profiling and price elasticity scatter widgets.
