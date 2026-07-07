import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def run_unified_pipeline():
    print("==================================================================")
    print("STARTING GRAND UNIFIED INTERACTIVE DASHBOARD PIPELINE")
    print("==================================================================")
    
    # 1. Paths
    base_dir = "d:/cat/DAL/project"
    input_file = os.path.join(base_dir, "data set/MasterFoodBeverage_Data.xlsx")
    dash_dir = os.path.join(base_dir, "InteractiveDashboard")
    output_dir = os.path.join(dash_dir, "cleaned_data")
    viz_dir = os.path.join(dash_dir, "visualizations")
    output_excel = os.path.join(dash_dir, "Cleaned_Restaurant_Data_Unified.xlsx")
    dashboard_file = os.path.join(dash_dir, "unified_dashboard.html")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # 2. Ingestion
    print(f"Reading source Excel file: {input_file}...")
    xls = pd.ExcelFile(input_file)
    sales_df = pd.read_excel(xls, 'Sales_Fact')
    customer_df = pd.read_excel(xls, 'Customer_Dim')
    menu_df = pd.read_excel(xls, 'Menu_Dim')
    category_df = pd.read_excel(xls, 'Category_Dim')
    location_df = pd.read_excel(xls, 'Location_Dim')
    date_df = pd.read_excel(xls, 'Date_Dim')
    
    # Clean strings
    for df in [sales_df, customer_df, menu_df, category_df, location_df, date_df]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])
    
    # 3. Clean & Correct Financial Columns (ETL Step)
    print("Correcting financial figures for 100% mathematical consistency...")
    sales_df['correct_gross_sales'] = sales_df['quantity'] * sales_df['unit_price']
    sales_df['correct_discount_amt'] = sales_df['correct_gross_sales'] * (sales_df['discount_pct'] / 100.0)
    sales_df['correct_net_sales'] = sales_df['correct_gross_sales'] - sales_df['correct_discount_amt']
    sales_df['correct_cogs'] = sales_df['quantity'] * sales_df['unit_cost']
    sales_df['correct_profit'] = sales_df['correct_net_sales'] - sales_df['correct_cogs']
    
    # Track original net sales and profit discrepancies for forensic audit
    sales_df['net_sales_diff'] = sales_df['correct_net_sales'] - sales_df['net_sales']
    sales_df['profit_diff'] = sales_df['correct_profit'] - sales_df['profit']
    
    # We round corrected columns
    for col in ['correct_gross_sales', 'correct_discount_amt', 'correct_net_sales', 'correct_cogs', 'correct_profit']:
        sales_df[col] = sales_df[col].round(2)
        
    # Denormalize Category_Dim into Menu_Dim
    dim_menu = menu_df.merge(category_df, on='category_id', how='left')
    dim_menu.rename(columns={'description': 'category_description'}, inplace=True)
    
    # Add age groups to Customer_Dim
    def get_age_group(age):
        if age < 25: return "<25"
        elif age <= 34: return "25-34"
        elif age <= 44: return "35-44"
        elif age <= 54: return "45-54"
        else: return "55+"
    customer_df['age_group'] = customer_df['age'].apply(get_age_group)
    
    # 4. Menu Engineering (BCG Matrix)
    print("Classifying menu items via BCG Matrix...")
    item_stats = sales_df.groupby('item_id').agg(
        total_quantity_sold=('quantity', 'sum'),
        total_net_sales=('correct_net_sales', 'sum'), # use correct net sales for menu engineering
        total_cogs=('correct_cogs', 'sum')
    ).reset_index()
    
    item_stats['unit_contribution_margin'] = (item_stats['total_net_sales'] - item_stats['total_cogs']) / item_stats['total_quantity_sold']
    
    avg_popularity = item_stats['total_quantity_sold'].mean()
    avg_profitability = item_stats['unit_contribution_margin'].mean()
    
    def classify_bcg(row):
        high_pop = row['total_quantity_sold'] >= avg_popularity
        high_profit = row['unit_contribution_margin'] >= avg_profitability
        if high_pop and high_profit: return 'Star'
        elif high_pop and not high_profit: return 'Plowhorse'
        elif not high_pop and high_profit: return 'Puzzle'
        else: return 'Dog'
        
    item_stats['menu_category'] = item_stats.apply(classify_bcg, axis=1)
    
    # Join BCG classification to dim_menu
    dim_menu = dim_menu.merge(item_stats[['item_id', 'total_quantity_sold', 'unit_contribution_margin', 'menu_category']], on='item_id', how='left')
    
    # 5. Customer Segmentation (RFM Analysis)
    print("Calculating RFM quintile scores and classifying customer behaviors...")
    ref_date = sales_df['order_date'].max()
    
    rfm_df = sales_df.groupby('customer_id').agg(
        last_purchase=('order_date', 'max'),
        Frequency=('order_id', 'nunique'),
        Monetary=('correct_net_sales', 'sum') # use corrected net sales
    ).reset_index()
    
    rfm_df['Recency'] = (ref_date - rfm_df['last_purchase']).dt.days
    
    # Quintiles
    rfm_df['Recency_Score'] = pd.qcut(rfm_df['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm_df['Frequency_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_df['Monetary_Score'] = pd.qcut(rfm_df['Monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    rfm_df['RFM_Group'] = rfm_df['Recency_Score'].astype(str) + rfm_df['Frequency_Score'].astype(str) + rfm_df['Monetary_Score'].astype(str)
    rfm_df['RFM_Score'] = rfm_df['Recency_Score'] + rfm_df['Frequency_Score'] + rfm_df['Monetary_Score']
    
    def classify_rfm(row):
        r = row['Recency_Score']
        f = row['Frequency_Score']
        m = row['Monetary_Score']
        
        if r >= 4 and f >= 4: return 'Champions'
        elif r >= 3 and f >= 3: return 'Loyal Customers'
        elif r >= 4 and f >= 2: return 'Potential Loyalists'
        elif r >= 4 and f == 1: return 'New Customers'
        elif r == 3 and f == 1: return 'Promising'
        elif r == 1 and f >= 4: return "Can't Lose Them"
        elif r in [1, 2] and f >= 3: return 'At Risk'
        elif r in [2, 3] and f in [2, 3]: return 'Need Attention'
        elif r in [2, 3] and f == 1: return 'About to Sleep'
        else: return 'Lost / Hibernating'
        
    rfm_df['Segment'] = rfm_df.apply(classify_rfm, axis=1)
    
    # Join RFM to Customer Dimension
    dim_customer_rfm = customer_df.merge(
        rfm_df[['customer_id', 'Recency', 'Frequency', 'Monetary', 
                'Recency_Score', 'Frequency_Score', 'Monetary_Score', 
                'RFM_Group', 'RFM_Score', 'Segment']], 
        on='customer_id', how='left'
    )
    
    # 6. Forensic Audit & Fraud Detection Flags
    print("Applying rule-based and statistical anomaly detection flags...")
    
    # Rules
    sales_df['flag_skimming'] = np.where((sales_df['quantity'] >= 20) & (sales_df['net_sales_diff'].abs() > 0.05), 1, 0)
    sales_df['flag_high_discount'] = np.where(sales_df['discount_pct'] >= 50, 1, 0)
    
    sales_price_check = sales_df.merge(menu_df[['item_id', 'base_price']], on='item_id', how='left')
    sales_df['price_ratio'] = sales_price_check['unit_price'] / sales_price_check['base_price']
    sales_df['flag_price_inflation'] = np.where(sales_df['price_ratio'] > 1.2, 1, 0)
    sales_df['flag_large_order'] = np.where((sales_df['quantity'] >= 20) & (sales_df['flag_skimming'] == 0), 1, 0)
    sales_df['flag_any_rule'] = np.where(
        (sales_df['flag_skimming'] == 1) | 
        (sales_df['flag_high_discount'] == 1) | 
        (sales_df['flag_price_inflation'] == 1) | 
        (sales_df['flag_large_order'] == 1), 1, 0
    )
    
    # Leakage
    sales_df['net_sales_leakage'] = np.where(sales_df['flag_skimming'] == 1, sales_df['net_sales_diff'], 0.0)
    sales_df['profit_leakage'] = np.where(sales_df['flag_skimming'] == 1, sales_df['profit_diff'], 0.0)
    
    # Statistical (Z-Score & IQR)
    net_sales_mean = sales_df['net_sales'].mean()
    net_sales_std = sales_df['net_sales'].std()
    sales_df['z_score_net_sales'] = (sales_df['net_sales'] - net_sales_mean) / net_sales_std
    sales_df['flag_zscore_outlier'] = np.where(sales_df['z_score_net_sales'].abs() > 3, 1, 0)
    
    q25 = sales_df['net_sales'].quantile(0.25)
    q75 = sales_df['net_sales'].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    sales_df['flag_iqr_outlier'] = np.where((sales_df['net_sales'] < lower_bound) | (sales_df['net_sales'] > upper_bound), 1, 0)
    
    # Benford's first digit
    def extract_first_digit(val):
        s = str(abs(val)).replace('.', '').lstrip('0')
        if len(s) > 0 and s[0] in '123456789':
            return int(s[0])
        return np.nan
    sales_df['first_digit'] = sales_df['net_sales'].apply(extract_first_digit)
    
    benford_probs = {d: np.log10(1 + 1/d) for d in range(1, 10)}
    branch_merged_temp = sales_df.merge(location_df, on='location_id')
    benford_results = {}
    for branch in branch_merged_temp['branch_name'].unique():
        branch_sales = branch_merged_temp[branch_merged_temp['branch_name'] == branch]
        digit_counts = branch_sales['first_digit'].value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
        mad = np.mean([abs(digit_counts[d] - benford_probs[d]) for d in range(1, 10)])
        status = "Non-Conformity" if mad >= 0.015 else ("Marginally Acceptable" if mad >= 0.012 else "Acceptable Conformity")
        risk = "High" if mad >= 0.015 else ("Medium" if mad >= 0.012 else "Low")
        benford_results[branch] = {
            "mad": float(mad),
            "status": status,
            "risk": risk,
            "distribution": digit_counts.to_dict()
        }
        
    # Unsupervised ML (Isolation Forest)
    features = ['quantity', 'unit_price', 'discount_pct', 'net_sales', 'cogs', 'profit', 'customer_rating']
    X = sales_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    sales_df['if_anomaly_pred'] = iso_forest.fit_predict(X_scaled)
    sales_df['flag_isolation_forest'] = np.where(sales_df['if_anomaly_pred'] == -1, 1, 0)
    sales_df['if_anomaly_score'] = iso_forest.decision_function(X_scaled)
    
    # 7. Package and Save Datasets
    print(f"Saving CSV outputs to {output_dir}...")
    
    sales_df.to_csv(os.path.join(output_dir, 'fact_sales_unified.csv'), index=False)
    dim_customer_rfm.to_csv(os.path.join(output_dir, 'dim_customer_unified.csv'), index=False)
    dim_menu.to_csv(os.path.join(output_dir, 'dim_menu_unified.csv'), index=False)
    location_df.to_csv(os.path.join(output_dir, 'dim_location_unified.csv'), index=False)
    date_df.to_csv(os.path.join(output_dir, 'dim_date_unified.csv'), index=False)
    
    print(f"Saving master Excel sheet: {output_excel}...")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        sales_df.to_excel(writer, sheet_name='fact_sales_unified', index=False)
        dim_customer_rfm.to_excel(writer, sheet_name='dim_customer_unified', index=False)
        dim_menu.to_excel(writer, sheet_name='dim_menu_unified', index=False)
        location_df.to_excel(writer, sheet_name='dim_location_unified', index=False)
        date_df.to_excel(writer, sheet_name='dim_date_unified', index=False)
        
    # 8. Generate Static Visualizations
    print("Generating static charts...")
    sns.set_theme(style="darkgrid")
    
    # 1. Monthly Trends
    plt.figure(figsize=(11, 5))
    monthly_stats = sales_df.groupby(['order_year', 'order_month']).agg(
        revenue=('correct_net_sales', 'sum'),
        profit=('correct_profit', 'sum')
    ).reset_index().sort_values(by=['order_year', 'order_month'])
    monthly_stats['month_label'] = monthly_stats.apply(lambda r: f"{int(r['order_year'])}-{int(r['order_month']):02d}", axis=1)
    plt.plot(monthly_stats['month_label'], monthly_stats['revenue'], marker='o', color='#3b82f6', label='Revenue', linewidth=2)
    plt.plot(monthly_stats['month_label'], monthly_stats['profit'], marker='s', color='#10b981', label='Profit', linewidth=2)
    plt.title('Monthly Financial Growth Trends (MoM)', fontsize=12, fontweight='bold', pad=10)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'unified_monthly_trend.png'), dpi=300)
    plt.close()
    
    # 2. BCG Matrix Scatter
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=dim_menu, x='total_quantity_sold', y='unit_contribution_margin', hue='menu_category', palette={'Star': '#10b981', 'Plowhorse': '#3b82f6', 'Puzzle': '#f59e0b', 'Dog': '#f43f5e'}, s=80)
    plt.axvline(x=avg_popularity, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=avg_profitability, color='gray', linestyle='--', linewidth=1)
    plt.title('BCG Menu Engineering Scatter Plot', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Total Quantity Sold')
    plt.ylabel('Unit Margin (Rs.)')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'unified_bcg_matrix.png'), dpi=300)
    plt.close()
    
    # 3. Customer Segments
    plt.figure(figsize=(10, 5))
    seg_counts = dim_customer_rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    sns.barplot(data=seg_counts, x='Count', y='Segment', palette='viridis', hue='Segment', legend=False)
    plt.title('RFM Behavioral Segment sizes', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'unified_rfm_segments.png'), dpi=300)
    plt.close()
    
    # 4. Branch Leakage
    plt.figure(figsize=(10, 5))
    branch_leak = sales_df.merge(location_df, on='location_id').groupby('branch_name')['net_sales_leakage'].sum().reset_index().sort_values(by='net_sales_leakage', ascending=False)
    sns.barplot(data=branch_leak, x='net_sales_leakage', y='branch_name', palette='Reds_r', hue='branch_name', legend=False)
    plt.title('Skimmed Revenue Leakage by Branch', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Leakage Amount (Rs.)')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'unified_branch_leakage.png'), dpi=300)
    plt.close()
    
    # 9. Compile Grand Unified HTML Dashboard
    print("Compiling Grand Unified HTML Dashboard...")
    
    # High-level KPIs JSON
    kpis = {
        'total_revenue': float(sales_df['correct_net_sales'].sum()),
        'total_profit': float(sales_df['correct_profit'].sum()),
        'overall_margin': float(sales_df['correct_profit'].sum() / sales_df['correct_net_sales'].sum() * 100),
        'total_orders': int(sales_df['order_id'].nunique()),
        'aov': float(sales_df['correct_net_sales'].sum() / sales_df['order_id'].nunique()),
        'avg_rating': float(sales_df['customer_rating'].mean()),
        'total_customers': int(len(dim_customer_rfm)),
        'total_leakage': float(sales_df['net_sales_leakage'].sum()),
        'total_leakage_profit': float(sales_df['profit_leakage'].sum()),
        'flagged_skimming': int(sales_df['flag_skimming'].sum()),
        'flagged_high_discount': int(sales_df['flag_high_discount'].sum()),
        'flagged_price_inflation': int(sales_df['flag_price_inflation'].sum()),
        'flagged_ml_outliers': int(sales_df['flag_isolation_forest'].sum())
    }
    
    # Monthly financial stats
    monthly_list = monthly_stats.to_dict(orient='records')
    
    # Category summary
    category_summary = sales_df.merge(dim_menu[['item_id', 'category_name']], on='item_id').groupby('category_name').agg(
        revenue=('correct_net_sales', 'sum'),
        profit=('correct_profit', 'sum')
    ).reset_index()
    category_summary['margin_pct'] = (category_summary['profit'] / category_summary['revenue'] * 100).round(2)
    category_list = category_summary.to_dict(orient='records')
    
    # Branch performance and utilization
    branch_merged = sales_df.merge(location_df, on='location_id')
    branch_perf = branch_merged.groupby('branch_name').agg(
        revenue=('correct_net_sales', 'sum'),
        profit=('correct_profit', 'sum'),
        orders=('order_id', 'nunique'),
        seating_cap=('seating_cap', 'first')
    ).reset_index()
    branch_perf['revenue_per_seat'] = (branch_perf['revenue'] / branch_perf['seating_cap']).round(2)
    branch_perf['profit_per_seat'] = (branch_perf['profit'] / branch_perf['seating_cap']).round(2)
    branch_list = branch_perf.sort_values(by='revenue', ascending=False).to_dict(orient='records')
    
    # BCG Matrix detailed items list
    menu_bcg_list = dim_menu[['item_name', 'category_name', 'total_quantity_sold', 'unit_contribution_margin', 'menu_category', 'base_price']].to_dict(orient='records')
    
    # Customer RFM Segments Summary
    seg_profiles = dim_customer_rfm.groupby('Segment').agg(
        customers=('customer_id', 'count'),
        avg_recency=('Recency', 'mean'),
        avg_frequency=('Frequency', 'mean'),
        avg_monetary=('Monetary', 'mean'),
        avg_loyalty=('loyalty_points', 'mean')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Search profiling sample (first 300 customers to keep HTML light)
    cust_lookup_data = dim_customer_rfm[['customer_id', 'customer_name', 'gender', 'age', 'customer_type', 'city', 'loyalty_points', 'Recency', 'Frequency', 'Monetary', 'RFM_Group', 'Segment']].head(300).round(2).to_dict(orient='records')
    
    # Quick High Spend Table
    top_spenders = dim_customer_rfm[['customer_name', 'city', 'loyalty_points', 'Frequency', 'Monetary', 'Segment']].sort_values(by='Monetary', ascending=False).head(25).round(2).to_dict(orient='records')
    
    # Fraud branch leakage summary
    branch_leak_agg = branch_merged.groupby('branch_name').agg(
        net_leakage=('net_sales_leakage', 'sum'),
        profit_leakage=('profit_leakage', 'sum'),
        skimming_count=('flag_skimming', 'sum')
    ).reset_index().to_dict(orient='records')
    
    # Top ML outliers (Isolation Forest)
    if_anoms = sales_df[sales_df['flag_isolation_forest'] == 1].sort_values(by='if_anomaly_score').head(20).merge(location_df, on='location_id')
    if_anoms_list = if_anoms[['order_id', 'branch_name', 'quantity', 'net_sales', 'correct_net_sales', 'profit', 'customer_rating', 'if_anomaly_score']].to_dict(orient='records')
    
    # Marketing Action Plan suggestions
    marketing_actions = {
        'Champions': 'Provide early menu access, VIP offers, and zero pricing friction. Upsell premium items.',
        'Loyal Customers': 'Inject loyalty point multiplier promotions, reward regular visits, recommend appetizers.',
        'Potential Loyalists': 'Provide multi-purchase discounts or starter+dessert bundle offers to increase visits.',
        'New Customers': 'Send an automated welcome voucher for their second visit within 14 days.',
        'Promising': 'Build awareness via email. Offer time-limited coupons for high-margin starters.',
        'Need Attention': 'Send time-limited discounts on their top-ordered items. Create meal bundles.',
        'About to Sleep': 'Trigger win-back sequences. Recommend weekend specials.',
        'At Risk': 'Offer a high-value discount coupon (e.g. Rs. 500 off) on their favorite meals.',
        "Can't Lose Them": 'Send a manager-signed invitation letter with a massive incentive (e.g. free dessert).',
        'Lost / Hibernating': 'Run low-cost double points campaigns or general brand re-engagement sweeps.'
    }
    
    # Combine data package
    dashboard_data = {
        'kpis': kpis,
        'monthly_stats': monthly_list,
        'category_stats': category_list,
        'branch_stats': branch_list,
        'menu_bcg': menu_bcg_list,
        'bcg_benchmarks': {
            'avg_qty': float(avg_popularity),
            'avg_margin': float(avg_profitability)
        },
        'rfm_profiles': seg_profiles,
        'cust_lookup': cust_lookup_data,
        'top_spenders': top_spenders,
        'marketing_actions': marketing_actions,
        'branch_leakage': branch_leak_agg,
        'benford_results': benford_results,
        'benford_theoretical': [benford_probs[d] for d in range(1, 10)],
        'if_anoms': if_anoms_list
    }
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Grand Unified Executive Dashboard</title>
    
    <!-- Outfit & Inter Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <script>
        window.onerror = function(msg, url, line, col, error) {{
            const errorDiv = document.createElement('div');
            errorDiv.style.position = 'fixed';
            errorDiv.style.top = '0';
            errorDiv.style.left = '0';
            errorDiv.style.width = '100%';
            errorDiv.style.backgroundColor = '#ef4444';
            errorDiv.style.color = '#fff';
            errorDiv.style.padding = '15px';
            errorDiv.style.zIndex = '99999';
            errorDiv.style.fontSize = '14px';
            errorDiv.style.fontFamily = 'monospace';
            errorDiv.style.borderBottom = '3px solid #b91c1c';
            errorDiv.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
            errorDiv.innerHTML = '<strong>JS Error:</strong> ' + msg + ' <br><small>in ' + url + ' on line ' + line + ':' + col + '</small>';
            document.body.appendChild(errorDiv);
            return false;
        }};
    </script>
    
    <style>
        :root {{
            --bg-base: #07090e;
            --bg-card: rgba(16, 18, 27, 0.75);
            --bg-card-hover: rgba(22, 25, 37, 0.95);
            --border-card: rgba(255, 255, 255, 0.05);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            
            --accent-blue: #3b82f6;
            --accent-teal: #0d9488;
            --accent-rose: #f43f5e;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-indigo: #6366f1;
            --accent-purple: #8b5cf6;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }}
        
        body {{
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.07) 0px, transparent 35%),
                radial-gradient(at 100% 100%, rgba(244, 63, 94, 0.06) 0px, transparent 35%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}
        
        h1, h2, h3, h4 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }}
        
        .glass-panel {{
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        }}
        
        .glass-panel:hover {{
            background: var(--bg-card-hover);
            border-color: rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
            background: rgba(16, 18, 27, 0.5);
            border-bottom: 1px solid var(--border-card);
            margin: -24px -24px 0 -24px;
            backdrop-filter: blur(8px);
            z-index: 10;
        }}
        
        header h1 {{
            font-size: 26px;
            background: linear-gradient(to right, #fff, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .tab-navigation {{
            display: flex;
            gap: 12px;
            margin-bottom: 8px;
        }}
        
        .tab-btn {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            padding: 10px 22px;
            border-radius: 9999px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        
        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.06);
        }}
        
        .tab-btn.active {{
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-blue);
            box-shadow: 0 0 16px rgba(59, 130, 246, 0.4);
        }}
        
        .dashboard-view {{
            display: none;
            flex-direction: column;
            gap: 24px;
        }}
        
        .dashboard-view.active {{
            display: flex;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 24px;
        }}
        
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
        }}
        
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }}
        
        .grid-6 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .kpi-card {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            position: relative;
            overflow: hidden;
        }}
        
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-blue);
        }}
        .kpi-card.emerald::before {{ background: var(--accent-emerald); }}
        .kpi-card.teal::before {{ background: var(--accent-teal); }}
        .kpi-card.rose::before {{ background: var(--accent-rose); }}
        .kpi-card.indigo::before {{ background: var(--accent-indigo); }}
        .kpi-card.purple::before {{ background: var(--accent-purple); }}
        
        .kpi-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .kpi-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 25px;
            font-weight: 800;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
            margin-top: 15px;
        }}
        
        .table-wrapper {{
            overflow-x: auto;
            border-radius: 8px;
            margin-top: 15px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 13px;
        }}
        
        th, td {{
            padding: 11px 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}
        
        th {{
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        
        tr:hover td {{
            background: rgba(255, 255, 255, 0.01);
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        
        /* BCG badges */
        .badge.star {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge.plowhorse {{ background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge.puzzle {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge.dog {{ background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.3); }}
        
        /* RFM badges */
        .badge.champions {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge.loyal_customers {{ background: rgba(139, 92, 246, 0.15); color: var(--accent-purple); border: 1px solid rgba(139, 92, 246, 0.3); }}
        .badge.potential_loyalists {{ background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge.new_customers {{ background: rgba(13, 148, 136, 0.15); color: var(--accent-teal); border: 1px solid rgba(13, 148, 136, 0.3); }}
        .badge.promising {{ background: rgba(234, 179, 8, 0.15); color: var(--accent-yellow); border: 1px solid rgba(234, 179, 8, 0.3); }}
        .badge.need_attention {{ background: rgba(249, 115, 22, 0.15); color: var(--accent-orange); border: 1px solid rgba(249, 115, 22, 0.3); }}
        .badge.about_to_sleep {{ background: rgba(156, 163, 175, 0.15); color: var(--text-secondary); border: 1px solid rgba(156, 163, 175, 0.3); }}
        .badge.at_risk {{ background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.3); }}
        .badge.cant_lose_them {{ background: rgba(219, 39, 119, 0.15); color: #db2777; border: 1px solid rgba(219, 39, 119, 0.3); }}
        .badge.lost_hibernating {{ background: rgba(75, 85, 99, 0.15); color: #9ca3af; border: 1px solid rgba(75, 85, 99, 0.3); }}

        /* Risk Badges */
        .badge.risk-high {{ background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.3); }}
        .badge.risk-med {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge.risk-low {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }}

        .action-item {{
            margin-top: 10px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid var(--accent-blue);
            border-radius: 4px;
            font-size: 13px;
        }}
        
        .action-title {{
            font-weight: 700;
            font-size: 13px;
            margin-bottom: 4px;
        }}

        .search-container {{
            display: flex;
            gap: 12px;
            margin-bottom: 15px;
        }}
        
        .search-input {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            color: white;
            padding: 10px 18px;
            font-size: 13px;
            outline: none;
            flex: 1;
        }}
        
        .search-input:focus {{
            border-color: var(--accent-blue);
        }}
        
        .profile-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 10px;
            font-size: 13px;
        }}
        
        .profile-label {{
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .profile-value {{
            font-weight: 700;
            text-align: right;
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Restaurant Unified Executive Dashboard</h1>
            <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">Grand Consolidated Command Center: Finance, Menu, Customers, & Fraud Audits (₹ Currency Model)</p>
        </div>
        <div style="font-size: 14px; text-align: right; color: var(--text-secondary);">
            <div>Executive Status: <span style="color: var(--accent-emerald); font-weight: 600;">Fully Consolidated</span></div>
            <div style="font-size: 11px; margin-top: 4px;">Unified DB: 52,541 rows mapped</div>
        </div>
    </header>

    <div class="tab-navigation">
        <button class="tab-btn active" onclick="switchTab('overview')">Executive Summary Hub</button>
        <button class="tab-btn" onclick="switchTab('menu')">Menu Engineering Hub</button>
        <button class="tab-btn" onclick="switchTab('customer')">Customer Behavior Hub</button>
        <button class="tab-btn" onclick="switchTab('audit')">Forensic Audit & Risk Hub</button>
    </div>

    <!-- VIEW 1: EXECUTIVE SUMMARY HUB -->
    <div id="view-overview" class="dashboard-view active">
        <!-- KPIs -->
        <div class="grid-6">
            <div class="glass-panel kpi-card teal">
                <div class="kpi-title">Correct Net Revenue</div>
                <div class="kpi-value" id="kpi-rev">₹0</div>
            </div>
            <div class="glass-panel kpi-card emerald">
                <div class="kpi-title">Correct Profit</div>
                <div class="kpi-value" id="kpi-profit">₹0</div>
            </div>
            <div class="glass-panel kpi-card teal">
                <div class="kpi-title">Profit Margin %</div>
                <div class="kpi-value" id="kpi-margin">0.0%</div>
            </div>
            <div class="glass-panel kpi-card rose">
                <div class="kpi-title">Total Leakage (Net)</div>
                <div class="kpi-value" id="kpi-leakage">₹0</div>
                <p id="kpi-leakage-pct" style="font-size: 9px; color: var(--text-secondary);">0.0% Leakage</p>
            </div>
            <div class="glass-panel kpi-card indigo">
                <div class="kpi-title">Unique Orders</div>
                <div class="kpi-value" id="kpi-orders">0</div>
            </div>
            <div class="glass-panel kpi-card purple">
                <div class="kpi-title">Audited Customers</div>
                <div class="kpi-value" id="kpi-customers">0</div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Monthly Trends -->
            <div class="glass-panel">
                <h3>Monthly Financial Performance Trends (MoM)</h3>
                <div class="chart-container">
                    <canvas id="chart-monthly-trend"></canvas>
                </div>
            </div>
            
            <!-- Category stack -->
            <div class="glass-panel">
                <h3>Category Sales & Margin Breakdown</h3>
                <div class="chart-container">
                    <canvas id="chart-category-breakdown"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Branch rank table -->
        <div class="glass-panel">
            <h3>Branch Efficiency Rankings & Seating Utilization</h3>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Branch Name</th>
                            <th>Total Revenue</th>
                            <th>Operating Profit</th>
                            <th>Total Orders</th>
                            <th>Seating Capacity</th>
                            <th>Revenue / Seat</th>
                            <th>Profit / Seat</th>
                        </tr>
                    </thead>
                    <tbody id="branch-table-tbody">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- VIEW 2: MENU ENGINEERING HUB -->
    <div id="view-menu" class="dashboard-view">
        <div class="grid-2">
            <!-- BCG Scatter -->
            <div class="glass-panel">
                <h3>BCG Matrix Menu Scatter Plot</h3>
                <p style="color: var(--text-secondary); font-size: 11px; margin-top: 4px;">Menu items mapped by popularity (total sold) vs profitability (unit contribution margin).</p>
                <div class="chart-container" style="height: 350px;">
                    <canvas id="chart-bcg-scatter"></canvas>
                </div>
            </div>
            
            <!-- BCG list -->
            <div class="glass-panel">
                <h3>Menu Engineering Detail</h3>
                <div style="display: flex; gap: 10px; margin-top: 10px; margin-bottom: 10px;">
                    <div style="padding: 8px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; flex: 1; text-align: center;">
                        <div style="font-size: 10px; color: var(--accent-emerald); font-weight:600; text-transform: uppercase;">Avg Qty Benchmark</div>
                        <div style="font-size: 18px; font-weight:800; font-family:'Outfit';" id="bcg-bench-qty">0</div>
                    </div>
                    <div style="padding: 8px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; flex: 1; text-align: center;">
                        <div style="font-size: 10px; color: var(--accent-blue); font-weight:600; text-transform: uppercase;">Avg Margin Benchmark</div>
                        <div style="font-size: 18px; font-weight:800; font-family:'Outfit';" id="bcg-bench-margin">₹0</div>
                    </div>
                </div>
                
                <div class="table-wrapper" style="max-height: 280px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Item Name</th>
                                <th>Category</th>
                                <th>Qty Sold</th>
                                <th>Unit Margin</th>
                                <th>BCG Segment</th>
                            </tr>
                        </thead>
                        <tbody id="bcg-table-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- VIEW 3: CUSTOMER BEHAVIOR HUB -->
    <div id="view-customer" class="dashboard-view">
        <div class="grid-2">
            <!-- RFM size distribution -->
            <div class="glass-panel">
                <h3>RFM Behavioral Segment Size Distribution</h3>
                <div class="chart-container">
                    <canvas id="chart-rfm-size"></canvas>
                </div>
            </div>
            
            <!-- Segment averages grid -->
            <div class="glass-panel">
                <h3>RFM Segment Profiling Grid</h3>
                <div class="table-wrapper" style="max-height: 300px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Segment Name</th>
                                <th>Customers</th>
                                <th>Avg Recency</th>
                                <th>Avg Freq</th>
                                <th>Avg Spend</th>
                                <th>Avg Loyalty</th>
                            </tr>
                        </thead>
                        <tbody id="rfm-profiles-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Search Lookup -->
            <div class="glass-panel">
                <h3>Customer Profiler Search</h3>
                <p style="color: var(--text-secondary); font-size: 12px; margin-bottom: 15px;">Search by customer ID or name to view their exact transactional metrics, RFM score, and segment.</p>
                
                <div class="search-container">
                    <input type="text" id="cust-search-input" class="search-input" placeholder="Enter customer ID or name (e.g. Customer_0001)..." oninput="searchCustomer()">
                </div>
                
                <div id="lookup-profile-card" style="display: none;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 10px; margin-bottom: 10px;">
                        <h4 id="profile-name" style="font-size: 16px; color: var(--accent-blue);">Customer Name</h4>
                        <span class="badge" id="profile-segment-badge">Segment</span>
                    </div>
                    
                    <div class="profile-grid">
                        <div><span class="profile-label">Customer ID:</span></div>
                        <div><span class="profile-value" id="profile-id">-</span></div>
                        
                        <div><span class="profile-label">City / Demographics:</span></div>
                        <div><span class="profile-value" id="profile-city">-</span></div>
                        
                        <div><span class="profile-label">Loyalty Points:</span></div>
                        <div><span class="profile-value" id="profile-loyalty" style="color: var(--accent-emerald);">-</span></div>
                        
                        <div><span class="profile-label">Recency Score:</span></div>
                        <div><span class="profile-value"><span id="profile-days">-</span> days (Score: <span id="profile-r-score">-</span>/5)</span></div>
                        
                        <div><span class="profile-label">Frequency Score:</span></div>
                        <div><span class="profile-value"><span id="profile-orders">-</span> orders (Score: <span id="profile-f-score">-</span>/5)</span></div>
                        
                        <div><span class="profile-label">Monetary Score:</span></div>
                        <div><span class="profile-value"><span id="profile-monetary">-</span> (Score: <span id="profile-m-score">-</span>/5)</span></div>
                        
                        <div><span class="profile-label">RFM Group Code:</span></div>
                        <div><span class="profile-value" id="profile-code" style="letter-spacing: 2px; color: var(--accent-purple);">-</span></div>
                    </div>
                    
                    <div class="action-item" style="margin-top: 15px;">
                        <div class="action-title">Targeted Engagement Advice</div>
                        <p id="profile-action-text">-</p>
                    </div>
                </div>
                
                <div id="lookup-search-help" style="color: var(--text-secondary); text-align: center; padding: 40px 0; font-size: 13px;">
                    Type a customer name above to query behavior profiles.
                </div>
            </div>
            
            <!-- Quick Top Spenders -->
            <div class="glass-panel">
                <h3>Top Revenue Spenders (Top 25)</h3>
                <div class="table-wrapper" style="max-height: 380px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>City</th>
                                <th>Orders</th>
                                <th>Spend</th>
                                <th>Segment</th>
                            </tr>
                        </thead>
                        <tbody id="top-spenders-body">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- VIEW 4: FORENSIC AUDIT & RISK HUB -->
    <div id="view-audit" class="dashboard-view">
        <!-- Audit KPIs -->
        <div class="grid-4">
            <div class="glass-panel kpi-card rose">
                <div class="kpi-title">Skimming Leakage (Profit)</div>
                <div class="kpi-value" id="kpi-leakage-prof">₹0</div>
                <p style="font-size: 9px; color: var(--text-secondary);" id="lbl-leakage-prof-pct">0.0% of operating profit</p>
            </div>
            <div class="glass-panel kpi-card rose">
                <div class="kpi-title">Revenue Skimming Cases</div>
                <div class="kpi-value" id="kpi-flagged-skimming">0</div>
                <p style="font-size: 9px; color: var(--text-secondary);">Quantity multipliers tampered</p>
            </div>
            <div class="glass-panel kpi-card amber">
                <div class="kpi-title">Sweethearting Cases</div>
                <div class="kpi-value" id="kpi-flagged-discount">0</div>
                <p style="font-size: 9px; color: var(--text-secondary);">Discounts >= 50%</p>
            </div>
            <div class="glass-panel kpi-card amber">
                <div class="kpi-title">Price Inflations</div>
                <div class="kpi-value" id="kpi-flagged-price">0</div>
                <p style="font-size: 9px; color: var(--text-secondary);">Overcharged > 20% compared to base</p>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Skimming leakage by branch -->
            <div class="glass-panel">
                <h3>Skimmed Revenue Leakage by Branch</h3>
                <p style="color: var(--text-secondary); font-size: 11px; margin-top: 4px;">Losses arising from bulk orders calculated on single-digit volumes.</p>
                <div class="chart-container">
                    <canvas id="chart-branch-leakage"></canvas>
                </div>
            </div>
            
            <!-- Benford line comparison -->
            <div class="glass-panel">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3>Benford's Law Digit Comparison</h3>
                    <div class="filter-group" style="display: flex; align-items: center; gap: 8px; flex-direction: row; font-size: 12px;">
                        <label for="benford-branch-select">Branch:</label>
                        <select id="benford-branch-select" onchange="updateBenfordChart()" style="background: var(--bg-base); color: white; border: 1px solid var(--border-card); padding: 4px 8px; border-radius: 4px;">
                            <!-- Populated by JS -->
                        </select>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-benford"></canvas>
                </div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- ML Outliers -->
            <div class="glass-panel">
                <h3>Top Multivariate ML Anomalies (Isolation Forest)</h3>
                <p style="color: var(--text-secondary); font-size: 12px; margin-top: 4px;">Isolation Forest multivariate outliers predicted using 7 joint transactional features.</p>
                <div class="table-wrapper" style="max-height: 250px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Order ID</th>
                                <th>Branch</th>
                                <th>Qty</th>
                                <th>Net Sales</th>
                                <th>Profit</th>
                                <th>Rating</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody id="ml-anoms-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Auditing Summary card -->
            <div class="glass-panel">
                <h3>Audit Risk evaluation</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Branch</th>
                                <th>Benford MAD</th>
                                <th>Rule 1 Cases</th>
                                <th>Audit Status</th>
                            </tr>
                        </thead>
                        <tbody id="audit-summary-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const unifiedData = {json.dumps(dashboard_data, indent=4)};
        
        // Tab switching
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.dashboard-view').forEach(view => view.classList.remove('active'));
            
            let activeBtn = document.querySelector(`.tab-btn[onclick="switchTab('${{tabId}}')"]`);
            if (activeBtn) activeBtn.classList.add('active');
            
            let targetView = document.getElementById(`view-${{tabId}}`);
            if (targetView) targetView.classList.add('active');

            // Force Chart.js to re-calculate dimensions for charts that were hidden on load
            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
            }}, 50);
        }}

        // Formatters
        const formatINR = (val) => {{
            return new Intl.NumberFormat('en-IN', {{ style: 'currency', currency: 'INR', maximumFractionDigits: 0 }}).format(val);
        }};
        const formatInt = (val) => {{
            return new Intl.NumberFormat('en-IN').format(val);
        }};

        // Set KPIs
        document.getElementById('kpi-rev').innerText = formatINR(unifiedData.kpis.total_revenue);
        document.getElementById('kpi-profit').innerText = formatINR(unifiedData.kpis.total_profit);
        document.getElementById('kpi-margin').innerText = unifiedData.kpis.overall_margin.toFixed(2) + "%";
        document.getElementById('kpi-orders').innerText = formatInt(unifiedData.kpis.total_orders);
        document.getElementById('kpi-customers').innerText = formatInt(unifiedData.kpis.total_customers);
        
        document.getElementById('kpi-leakage').innerText = formatINR(unifiedData.kpis.total_leakage);
        document.getElementById('kpi-leakage-pct').innerText = (unifiedData.kpis.total_leakage / unifiedData.kpis.total_revenue * 100).toFixed(2) + "% of revenue underreported";
        document.getElementById('kpi-leakage-prof').innerText = formatINR(unifiedData.kpis.total_leakage_profit);
        document.getElementById('lbl-leakage-prof-pct').innerText = (unifiedData.kpis.total_leakage_profit / unifiedData.kpis.total_profit * 100).toFixed(2) + "% of operating profit";
        
        document.getElementById('kpi-flagged-skimming').innerText = formatInt(unifiedData.kpis.flagged_skimming);
        document.getElementById('kpi-flagged-discount').innerText = formatInt(unifiedData.kpis.flagged_high_discount);
        document.getElementById('kpi-flagged-price').innerText = formatInt(unifiedData.kpis.flagged_price_inflation);

        document.getElementById('bcg-bench-qty').innerText = formatInt(unifiedData.bcg_benchmarks.avg_qty);
        document.getElementById('bcg-bench-margin').innerText = formatINR(unifiedData.bcg_benchmarks.avg_margin);

        // Populate Branch Table
        const branchTbody = document.getElementById('branch-table-tbody');
        unifiedData.branch_stats.forEach(b => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600;">${{b.branch_name}}</td>
                <td style="font-weight:600;">${{formatINR(b.revenue)}}</td>
                <td style="color: var(--accent-emerald); font-weight:600;">${{formatINR(b.profit)}}</td>
                <td>${{formatInt(b.orders)}}</td>
                <td>${{b.seating_cap}} seats</td>
                <td>${{formatINR(b.revenue_per_seat)}} / seat</td>
                <td style="font-weight:600; color: var(--accent-teal);">${{formatINR(b.profit_per_seat)}} / seat</td>
            `;
            branchTbody.appendChild(tr);
        }});

        // Populate BCG Table
        const bcgTbody = document.getElementById('bcg-table-tbody');
        unifiedData.menu_bcg.forEach(item => {{
            const tr = document.createElement('tr');
            let badgeClass = 'dog';
            if (item.menu_category === 'Star') badgeClass = 'star';
            else if (item.menu_category === 'Plowhorse') badgeClass = 'plowhorse';
            else if (item.menu_category === 'Puzzle') badgeClass = 'puzzle';
            
            tr.innerHTML = `
                <td style="font-weight:600;">${{item.item_name}}</td>
                <td>${{item.category_name}}</td>
                <td>${{formatInt(item.total_quantity_sold)}}</td>
                <td style="font-weight:600;">${{formatINR(item.unit_contribution_margin)}}</td>
                <td><span class="badge ${{badgeClass}}">${{item.menu_category}}</span></td>
            `;
            bcgTbody.appendChild(tr);
        }});

        // Populate RFM Profiles Table
        const rfmProfilesTbody = document.getElementById('rfm-profiles-tbody');
        unifiedData.rfm_profiles.forEach(p => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="badge ${{p.Segment.toLowerCase().replace("'", "").replace("/ ", "").replace(" ", "_")}}">${{p.Segment}}</span></td>
                <td style="font-weight:600;">${{formatInt(p.customers)}}</td>
                <td>${{p.avg_recency.toFixed(1)}}</td>
                <td>${{p.avg_frequency.toFixed(2)}}</td>
                <td style="font-weight:600;">${{formatINR(p.avg_monetary)}}</td>
                <td style="color: var(--accent-emerald)">${{formatInt(Math.round(p.avg_loyalty))}}</td>
            `;
            rfmProfilesTbody.appendChild(tr);
        }});

        // Populate Top Spenders Table
        const spenderTbody = document.getElementById('top-spenders-body');
        unifiedData.top_spenders.forEach(c => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600; cursor:pointer; color: var(--accent-blue);" onclick="fillSearch('${{c.customer_name}}')">${{c.customer_name}}</td>
                <td>${{c.city}}</td>
                <td>${{c.Frequency}}</td>
                <td style="font-weight:600;">${{formatINR(c.Monetary)}}</td>
                <td><span class="badge ${{c.Segment.toLowerCase().replace("'", "").replace("/ ", "").replace(" ", "_")}}">${{c.Segment}}</span></td>
            `;
            spenderTbody.appendChild(tr);
        }});

        // Populate ML Anomalies Table
        const mlTbody = document.getElementById('ml-anoms-tbody');
        unifiedData.if_anoms.forEach(anom => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600;">#${{anom.order_id}}</td>
                <td>${{anom.branch_name}}</td>
                <td>${{anom.quantity}}</td>
                <td>${{formatINR(anom.net_sales)}}</td>
                <td style="color: var(--accent-rose); font-weight:600;">${{formatINR(anom.profit)}}</td>
                <td>${{anom.customer_rating}} ⭐</td>
                <td style="color: var(--accent-rose); font-weight:600;">${{anom.if_anomaly_score.toFixed(4)}}</td>
            `;
            mlTbody.appendChild(tr);
        }});

        // Populate Auditing Summary Table
        const auditSummaryTbody = document.getElementById('audit-summary-tbody');
        const branchSelect = document.getElementById('benford-branch-select');
        
        unifiedData.branch_leakage.forEach((b, idx) => {{
            const benford = unifiedData.benford_results[b.branch_name];
            const tr = document.createElement('tr');
            
            let badgeClass = 'risk-low';
            if (benford.risk === 'High') badgeClass = 'risk-high';
            else if (benford.risk === 'Medium') badgeClass = 'risk-med';
            
            tr.innerHTML = `
                <td style="font-weight:600;">${{b.branch_name}}</td>
                <td>${{benford.mad.toFixed(4)}}</td>
                <td style="font-weight:600; color: var(--accent-rose);">${{b.skimming_count}}</td>
                <td><span class="badge ${{badgeClass}}">${{benford.status}}</span></td>
            `;
            auditSummaryTbody.appendChild(tr);
            
            // Populate branch dropdown
            const opt = document.createElement('option');
            opt.value = b.branch_name;
            opt.innerText = b.branch_name;
            if (idx === 0) opt.selected = true;
            branchSelect.appendChild(opt);
        }});

        // Customer Search Logic
        function fillSearch(name) {{
            document.getElementById('cust-search-input').value = name;
            switchTab('customer');
            searchCustomer();
        }}
        
        function searchCustomer() {{
            const val = document.getElementById('cust-search-input').value.toLowerCase().trim();
            const card = document.getElementById('lookup-profile-card');
            const help = document.getElementById('lookup-search-help');
            
            if (val === "") {{
                card.style.display = "none";
                help.style.display = "";
                return;
            }}
            
            // Find customer
            const cust = unifiedData.cust_lookup.find(c => c.customer_name.toLowerCase().includes(val) || c.customer_id.toString().includes(val));
            
            if (cust) {{
                card.style.display = "";
                help.style.display = "none";
                
                document.getElementById('profile-name').innerText = cust.customer_name;
                document.getElementById('profile-id').innerText = "#" + cust.customer_id;
                document.getElementById('profile-city').innerText = cust.city + " (" + cust.gender + ", age " + cust.age + ")";
                document.getElementById('profile-loyalty').innerText = formatInt(cust.loyalty_points) + " pts";
                
                document.getElementById('profile-segment-badge').innerText = cust.Segment;
                document.getElementById('profile-segment-badge').className = "badge " + cust.Segment.toLowerCase().replace("'", "").replace("/ ", "").replace(" ", "_");
                
                document.getElementById('profile-days').innerText = cust.Recency;
                document.getElementById('profile-r-score').innerText = cust.Recency_Score;
                document.getElementById('profile-orders').innerText = cust.Frequency;
                document.getElementById('profile-f-score').innerText = cust.Frequency_Score;
                document.getElementById('profile-monetary').innerText = formatINR(cust.Monetary);
                document.getElementById('profile-m-score').innerText = cust.Monetary_Score;
                document.getElementById('profile-code').innerText = cust.RFM_Group;
                
                // Set advice
                document.getElementById('profile-action-text').innerText = unifiedData.marketing_actions[cust.Segment];
            }} else {{
                card.style.display = "none";
                help.style.display = "";
                help.innerText = "No customer found matching '" + val + "'. (Lookup limited to first 300 customers)";
            }}
        }}

        // ----------------------------------------------------
        // CHARTS SETUP (Chart.js)
        // ----------------------------------------------------
        
        const segmentColors = {{
            'Star': '#10b981',      // emerald
            'Plowhorse': '#3b82f6', // blue
            'Puzzle': '#f59e0b',    // amber
            'Dog': '#ef4444'        // red
        }};
        
        const segmentColors = {{
            'Star': '#10b981',      // emerald
            'Plowhorse': '#3b82f6', // blue
            'Puzzle': '#f59e0b',    // amber
            'Dog': '#ef4444'        // red
        }};
        
        const segColors = {{
            'Champions': '#10b981',
            'Loyal Customers': '#3b82f6',
            'Potential Loyalists': '#06b6d4',
            'New Customers': '#6366f1',
            'Promising': '#8b5cf6',
            'Need Attention': '#f59e0b',
            'About to Sleep': '#f97316',
            'At Risk': '#ef4444',
            "Can't Lose Them": '#e11d48',
            'Lost / Hibernating': '#9ca3af'
        }};
        
        // 1. Monthly Trends Chart
        const ctxMonthly = document.getElementById('chart-monthly-trend').getContext('2d');
        const months = unifiedData.monthly_stats.map(m => m.month_label);
        const revenues = unifiedData.monthly_stats.map(m => m.revenue);
        const profits = unifiedData.monthly_stats.map(m => m.profit);
        
        new Chart(ctxMonthly, {{
            type: 'line',
            data: {{
                labels: months,
                datasets: [
                    {{
                        label: 'Net Sales',
                        data: revenues,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        fill: true
                    }},
                    {{
                        label: 'Operating Profit',
                        data: profits,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        fill: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#f3f4f6' }} }} }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#9ca3af' }} }},
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }} }}
                }}
            }}
        }});

        // 2. Category Breakdown Chart
        const ctxCategory = document.getElementById('chart-category-breakdown').getContext('2d');
        new Chart(ctxCategory, {{
            type: 'bar',
            data: {{
                labels: unifiedData.category_stats.map(c => c.category_name),
                datasets: [
                    {{
                        label: 'Net Sales',
                        data: unifiedData.category_stats.map(c => c.revenue),
                        backgroundColor: 'rgba(99, 102, 241, 0.85)',
                        borderColor: '#6366f1',
                        borderRadius: 6,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Margin %',
                        data: unifiedData.category_stats.map(c => c.margin_pct),
                        type: 'line',
                        borderColor: '#eab308',
                        borderWidth: 2,
                        pointBackgroundColor: '#eab308',
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#f3f4f6' }} }} }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#9ca3af' }} }},
                    y: {{
                        type: 'linear',
                        position: 'left',
                        grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                        ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }}
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#9ca3af', callback: function(value) {{ return value + '%'; }} }}
                    }}
                }}
            }}
        }});

        // 3. BCG Scatter
        const ctxBCG = document.getElementById('chart-bcg-scatter').getContext('2d');
        const bcgPoints = unifiedData.menu_bcg.map(item => ({{
            x: item.total_quantity_sold,
            y: item.unit_contribution_margin,
            label: item.item_name,
            segment: item.menu_category
        }}));
        new Chart(ctxBCG, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    data: bcgPoints,
                    pointBackgroundColor: bcgPoints.map(p => segmentColors[p.segment] || '#3b82f6'),
                    pointBorderColor: 'rgba(0,0,0,0.5)',
                    pointRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const p = context.raw;
                                return `${{p.label}} (${{p.segment}}): Sold ${{formatInt(p.x)}} | Margin: ${{formatINR(p.y)}}`;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }},
                    y: {{ ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
                }}
            }}
        }});

        // 4. RFM Size
        const ctxRFMSize = document.getElementById('chart-rfm-size').getContext('2d');
        new Chart(ctxRFMSize, {{
            type: 'bar',
            data: {{
                labels: unifiedData.rfm_profiles.map(p => p.Segment),
                datasets: [{{
                    data: unifiedData.rfm_profiles.map(p => p.customers),
                    backgroundColor: unifiedData.rfm_profiles.map(p => segColors[p.Segment] || '#9ca3af'),
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ display: false }} }},
                    y: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
                }}
            }}
        }});

        // 5. Branch Leakage Chart
        const ctxBranchLeak = document.getElementById('chart-branch-leakage').getContext('2d');
        new Chart(ctxBranchLeak, {{
            type: 'bar',
            data: {{
                labels: unifiedData.branch_leakage.map(b => b.branch_name),
                datasets: [
                    {{
                        label: 'Net Sales Leakage',
                        data: unifiedData.branch_leakage.map(b => b.net_leakage),
                        backgroundColor: 'rgba(244, 63, 94, 0.85)',
                        borderColor: '#f43f5e',
                        borderRadius: 4,
                        borderWidth: 1
                    }},
                    {{
                        label: 'Profit Leakage',
                        data: unifiedData.branch_leakage.map(b => b.profit_leakage),
                        backgroundColor: 'rgba(245, 158, 11, 0.85)',
                        borderColor: '#f59e0b',
                        borderRadius: 4,
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#f3f4f6' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
                    y: {{ ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }}
                }}
            }}
        }});

        // 6. Benford comparison
        const ctxBenford = document.getElementById('chart-benford').getContext('2d');
        let benfordChart;
        function drawBenfordChart(branchName) {{
            const actualProportions = [];
            const labels = [];
            for (let d = 1; d <= 9; d++) {{
                labels.push(d.toString());
                actualProportions.push(unifiedData.benford_results[branchName].distribution[d.toString()]);
            }}
            if (benfordChart) benfordChart.destroy();
            benfordChart = new Chart(ctxBenford, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            type: 'line',
                            label: "Benford Theoretical",
                            data: unifiedData.benford_theoretical,
                            borderColor: '#3b82f6',
                            borderWidth: 2,
                            fill: false
                        }},
                        {{
                            label: "Actual Proportion",
                            data: actualProportions,
                            backgroundColor: 'rgba(244, 63, 94, 0.65)',
                            borderColor: '#f43f5e',
                            borderRadius: 4,
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ labels: {{ color: '#f3f4f6' }} }} }},
                    scales: {{
                        x: {{ ticks: {{ color: '#9ca3af' }} }},
                        y: {{ ticks: {{ color: '#9ca3af', callback: function(v) {{ return (v*100).toFixed(0) + '%'; }} }} }}
                    }}
                }}
            }});
        }}
        const firstBranch = Object.keys(unifiedData.benford_results)[0];
        drawBenfordChart(firstBranch);
        function updateBenfordChart() {{
            const selected = document.getElementById('benford-branch-select').value;
            drawBenfordChart(selected);
        }}
        
    </script>
</body>
</html>
"""
    
    print(f"Writing interactive HTML dashboard to {dashboard_file}...")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    print("==================================================================")
    print("GRAND UNIFIED PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================================")

if __name__ == "__main__":
    run_unified_pipeline()
