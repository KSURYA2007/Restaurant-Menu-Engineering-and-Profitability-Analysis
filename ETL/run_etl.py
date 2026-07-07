import os
import pandas as pd
import numpy as np

def run_etl():
    print("Starting ETL Process...")
    
    # 1. Paths
    input_file = "d:/cat/DAL/project/data set/MasterFoodBeverage_Data.xlsx"
    output_dir = "d:/cat/DAL/project/ETL/cleaned_data"
    output_excel = "d:/cat/DAL/project/ETL/Cleaned_Restaurant_Data.xlsx"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Extract
    print(f"Reading source data from {input_file}...")
    xls = pd.ExcelFile(input_file)
    sales_df = pd.read_excel(xls, 'Sales_Fact')
    customer_df = pd.read_excel(xls, 'Customer_Dim')
    menu_df = pd.read_excel(xls, 'Menu_Dim')
    category_df = pd.read_excel(xls, 'Category_Dim')
    location_df = pd.read_excel(xls, 'Location_Dim')
    date_df = pd.read_excel(xls, 'Date_Dim')
    
    # 3. Transform Dimensions
    print("Transforming Dimension Tables...")
    
    # Customer Dimension Clean
    customer_clean = customer_df.copy()
    for col in ['customer_name', 'gender', 'customer_type', 'city', 'email', 'phone']:
        customer_clean[col] = customer_clean[col].astype(str).str.strip()
    customer_clean['join_date'] = pd.to_datetime(customer_clean['join_date'])
    
    # Location Dimension Clean
    location_clean = location_df.copy()
    for col in ['branch_name', 'city', 'region']:
        location_clean[col] = location_clean[col].astype(str).str.strip()
    location_clean['opened_date'] = pd.to_datetime(location_clean['opened_date'])
    
    # Date Dimension Clean
    date_clean = date_df.copy()
    date_clean['date'] = pd.to_datetime(date_clean['date'])
    date_clean['month_name'] = date_clean['month_name'].astype(str).str.strip()
    date_clean['day_name'] = date_clean['day_name'].astype(str).str.strip()
    
    # Menu Dimension Clean & Star Schema Denormalization
    # Denormalize Category_Dim into Menu_Dim to make it a pure Star Schema (flat menu dimension)
    print("Denormalizing Category into Menu Dimension for Star Schema design...")
    menu_clean = menu_df.copy()
    for col in ['item_name', 'is_vegetarian']:
        menu_clean[col] = menu_clean[col].astype(str).str.strip()
        
    category_clean = category_df.copy()
    for col in ['category_name', 'description']:
        category_clean[col] = category_clean[col].astype(str).str.strip()
        
    dim_menu = menu_clean.merge(category_clean, on='category_id', how='left')
    dim_menu.rename(columns={'description': 'category_description'}, inplace=True)
    
    # 4. Transform Fact Table (Sales_Fact)
    print("Transforming Fact Table and Correcting Financial Columns...")
    fact_sales = sales_df.copy()
    
    # Ensure correct datatypes
    fact_sales['order_date'] = pd.to_datetime(fact_sales['order_date'])
    for col in ['payment_method', 'order_type']:
        fact_sales[col] = fact_sales[col].astype(str).str.strip()
        
    # Recalculate financial fields for 100% mathematical consistency
    # Calculations rules:
    # 1. gross_sales = quantity * unit_price
    # 2. discount_amt = gross_sales * (discount_pct / 100)
    # 3. net_sales = gross_sales - discount_amt
    # 4. cogs = quantity * unit_cost
    # 5. profit = net_sales - cogs
    # 6. profit_margin = profit / net_sales
    
    print("Recalculating financial figures...")
    fact_sales['gross_sales'] = fact_sales['quantity'] * fact_sales['unit_price']
    fact_sales['discount_amt'] = fact_sales['gross_sales'] * (fact_sales['discount_pct'] / 100.0)
    fact_sales['net_sales'] = fact_sales['gross_sales'] - fact_sales['discount_amt']
    fact_sales['cogs'] = fact_sales['quantity'] * fact_sales['unit_cost']
    fact_sales['profit'] = fact_sales['net_sales'] - fact_sales['cogs']
    
    # Handle division by zero for profit_margin in case net_sales is 0 (though min net_sales was 11.51)
    fact_sales['profit_margin'] = np.where(fact_sales['net_sales'] != 0, fact_sales['profit'] / fact_sales['net_sales'], 0.0)
    
    # Round financial columns to 2 decimal places
    financial_cols = ['gross_sales', 'discount_amt', 'net_sales', 'cogs', 'profit', 'profit_margin']
    for col in financial_cols:
        if col == 'profit_margin':
            # We keep it as a fraction (e.g. 0.7107) but round to 4 decimals for high precision
            fact_sales[col] = fact_sales[col].round(4)
        else:
            fact_sales[col] = fact_sales[col].round(2)
            
    # 5. Menu Engineering (BCG Matrix Classification)
    print("Performing Menu Engineering analysis...")
    # Calculate performance metrics for each menu item from the fact table
    item_stats = fact_sales.groupby('item_id').agg(
        total_quantity_sold=('quantity', 'sum'),
        total_net_sales=('net_sales', 'sum'),
        total_cogs=('cogs', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()
    
    # Unit contribution margin = (total net sales - total cogs) / total quantity sold
    # which is also equal to weighted unit profit margin
    item_stats['unit_contribution_margin'] = (item_stats['total_net_sales'] - item_stats['total_cogs']) / item_stats['total_quantity_sold']
    
    # Benchmarks
    avg_popularity = item_stats['total_quantity_sold'].mean()
    avg_profitability = item_stats['unit_contribution_margin'].mean()
    
    print(f"Popularity Benchmark (Average Qty Sold): {avg_popularity:.2f}")
    print(f"Profitability Benchmark (Average Unit Contribution Margin): {avg_profitability:.2f}")
    
    # Classify items
    def classify_bcg(row):
        high_pop = row['total_quantity_sold'] >= avg_popularity
        high_profit = row['unit_contribution_margin'] >= avg_profitability
        if high_pop and high_profit:
            return 'Star'
        elif high_pop and not high_profit:
            return 'Plowhorse'
        elif not high_pop and high_profit:
            return 'Puzzle'
        else:
            return 'Dog'
            
    item_stats['menu_category'] = item_stats.apply(classify_bcg, axis=1)
    
    # Join the classifications back to dim_menu
    dim_menu = dim_menu.merge(item_stats[['item_id', 'total_quantity_sold', 'unit_contribution_margin', 'menu_category']], on='item_id', how='left')
    
    # 6. Load - Save cleaned dataset
    print(f"Saving cleaned dataset files to {output_dir}...")
    fact_sales.to_csv(os.path.join(output_dir, 'fact_sales.csv'), index=False)
    customer_clean.to_csv(os.path.join(output_dir, 'dim_customer.csv'), index=False)
    dim_menu.to_csv(os.path.join(output_dir, 'dim_menu.csv'), index=False)
    location_clean.to_csv(os.path.join(output_dir, 'dim_location.csv'), index=False)
    date_clean.to_csv(os.path.join(output_dir, 'dim_date.csv'), index=False)
    
    print(f"Writing master Excel sheet with all tables to {output_excel}...")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        fact_sales.to_excel(writer, sheet_name='fact_sales', index=False)
        customer_clean.to_excel(writer, sheet_name='dim_customer', index=False)
        dim_menu.to_excel(writer, sheet_name='dim_menu', index=False)
        location_clean.to_excel(writer, sheet_name='dim_location', index=False)
        date_clean.to_excel(writer, sheet_name='dim_date', index=False)
        
    print("ETL Process Completed Successfully!")
    
    # 7. Print validation metrics
    print("\n--- ETL Validation Checks ---")
    print(f"Fact Table rows: {len(fact_sales)} (Original: {len(sales_df)})")
    print(f"Customer Dim rows: {len(customer_clean)} (Original: {len(customer_df)})")
    print(f"Menu Dim rows: {len(dim_menu)} (Original: {len(menu_df)})")
    print(f"Location Dim rows: {len(location_clean)} (Original: {len(location_df)})")
    print(f"Date Dim rows: {len(date_clean)} (Original: {len(date_df)})")
    
    # Calculation checks
    diff_profit = np.abs(fact_sales['profit'] - (fact_sales['net_sales'] - fact_sales['cogs'])).max()
    print(f"Max absolute profit validation difference: {diff_profit:.6f}")
    
    # BCG Distribution
    print("\nBCG Matrix Distribution in Menu Items:")
    print(dim_menu['menu_category'].value_counts())

if __name__ == "__main__":
    run_etl()
