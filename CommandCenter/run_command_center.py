import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_command_center():
    print("==================================================================")
    print("STARTING KEY KPI MONITORING DASHBOARD (COMMAND CENTER) PIPELINE")
    print("==================================================================")
    
    # 1. Paths
    base_dir = "d:/cat/DAL/project"
    input_file = os.path.join(base_dir, "data set/MasterFoodBeverage_Data.xlsx")
    cmd_dir = os.path.join(base_dir, "CommandCenter")
    output_dir = os.path.join(cmd_dir, "cleaned_data")
    viz_dir = os.path.join(cmd_dir, "visualizations")
    output_excel = os.path.join(cmd_dir, "Cleaned_Restaurant_Data_CommandCenter.xlsx")
    dashboard_file = os.path.join(cmd_dir, "command_center.html")
    
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
                
    # 3. Clean & Transform Dimensions & Recalculate Fact (100% Mathematical Consistency)
    print("Pre-processing and cleaning data sheets...")
    
    # Recalculate sales fact to ensure mathematical accuracy
    sales_df['gross_sales'] = sales_df['quantity'] * sales_df['unit_price']
    sales_df['discount_amt'] = sales_df['gross_sales'] * (sales_df['discount_pct'] / 100.0)
    sales_df['net_sales'] = sales_df['gross_sales'] - sales_df['discount_amt']
    sales_df['cogs'] = sales_df['quantity'] * sales_df['unit_cost']
    sales_df['profit'] = sales_df['net_sales'] - sales_df['cogs']
    sales_df['profit_margin'] = np.where(sales_df['net_sales'] != 0, sales_df['profit'] / sales_df['net_sales'] * 100, 0.0)
    
    # Round financial columns
    financial_cols = ['gross_sales', 'discount_amt', 'net_sales', 'cogs', 'profit', 'profit_margin']
    for col in financial_cols:
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
    
    # 4. Strategic Analysis: Menu Engineering (BCG Matrix)
    print("Performing Menu Engineering (BCG Matrix) analysis...")
    item_stats = sales_df.groupby('item_id').agg(
        total_quantity_sold=('quantity', 'sum'),
        total_net_sales=('net_sales', 'sum'),
        total_cogs=('cogs', 'sum')
    ).reset_index()
    
    # Unit contribution margin = (total net sales - total cogs) / total quantity sold
    item_stats['unit_contribution_margin'] = (item_stats['total_net_sales'] - item_stats['total_cogs']) / item_stats['total_quantity_sold']
    
    # Benchmarks
    avg_popularity = item_stats['total_quantity_sold'].mean()
    avg_profitability = item_stats['unit_contribution_margin'].mean()
    
    print(f"Popularity Benchmark (Avg Quantity Sold): {avg_popularity:.2f}")
    print(f"Profitability Benchmark (Avg Unit Margin): {avg_profitability:.2f}")
    
    # Classify items
    def classify_bcg(row):
        high_pop = row['total_quantity_sold'] >= avg_popularity
        high_profit = row['unit_contribution_margin'] >= avg_profitability
        if high_pop and high_profit: return 'Star'
        elif high_pop and not high_profit: return 'Plowhorse'
        elif not high_pop and high_profit: return 'Puzzle'
        else: return 'Dog'
        
    item_stats['menu_category'] = item_stats.apply(classify_bcg, axis=1)
    
    # Join BCG classification back to Menu Dim
    dim_menu = dim_menu.merge(item_stats[['item_id', 'total_quantity_sold', 'unit_contribution_margin', 'menu_category']], on='item_id', how='left')
    
    # 5. Strategic Analysis: Time-Series Trends (Monthly Revenue & Profit)
    print("Compiling monthly financial trends...")
    # Group by year and month
    monthly_stats = sales_df.groupby(['order_year', 'order_month']).agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        orders=('order_id', 'nunique'),
        quantity_sold=('quantity', 'sum')
    ).reset_index().sort_values(by=['order_year', 'order_month'])
    
    # Format a Month label like "2022-01", "2022-02", etc.
    monthly_stats['month_label'] = monthly_stats.apply(lambda r: f"{int(r['order_year'])}-{int(r['order_month']):02d}", axis=1)
    
    # MoM Growth
    monthly_stats['revenue_mom_growth'] = monthly_stats['revenue'].pct_change() * 100
    monthly_stats['profit_mom_growth'] = monthly_stats['profit'].pct_change() * 100
    monthly_stats = monthly_stats.round(2)
    
    # 6. Operational Analysis: Channel splits, Branch rankings
    print("Compiling operational metrics...")
    
    # Category summary
    category_summary = sales_df.merge(dim_menu[['item_id', 'category_name']], on='item_id').groupby('category_name').agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        quantity=('quantity', 'sum')
    ).reset_index()
    category_summary['margin_pct'] = (category_summary['profit'] / category_summary['revenue'] * 100).round(2)
    
    # Branch performance and seating capacity utilization
    branch_merged = sales_df.merge(location_df, on='location_id')
    branch_perf = branch_merged.groupby('branch_name').agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        orders=('order_id', 'nunique'),
        seating_cap=('seating_cap', 'first')
    ).reset_index()
    branch_perf['revenue_per_seat'] = (branch_perf['revenue'] / branch_perf['seating_cap']).round(2)
    branch_perf['profit_per_seat'] = (branch_perf['profit'] / branch_perf['seating_cap']).round(2)
    branch_perf = branch_perf.sort_values(by='revenue', ascending=False)
    
    # Day of Week order distribution
    # order_dayofweek: 0 = Mon, ..., 6 = Sun
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    sales_df['day_name'] = sales_df['order_dayofweek'].map(day_mapping)
    day_perf = sales_df.groupby('order_dayofweek').agg(
        orders=('order_id', 'nunique'),
        revenue=('net_sales', 'sum')
    ).reset_index()
    day_perf['day_name'] = day_perf['order_dayofweek'].map(day_mapping)
    day_perf = day_perf.sort_values(by='order_dayofweek')
    
    # Channel splits
    order_type_perf = sales_df.groupby('order_type')['net_sales'].sum().reset_index()
    payment_perf = sales_df.groupby('payment_method')['net_sales'].sum().reset_index()
    
    # 7. Customer Insights Analysis
    print("Compiling customer demographics and loyalty stats...")
    sales_cust = sales_df.merge(customer_df, on='customer_id')
    
    customer_type_perf = sales_cust.groupby('customer_type').agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        customers=('customer_id', 'nunique')
    ).reset_index()
    
    age_group_perf = sales_cust.groupby('age_group').agg(
        revenue=('net_sales', 'sum'),
        orders=('order_id', 'nunique')
    ).reset_index().sort_values(by='age_group')
    
    # Loyalty correlation sample (top 200 customers to avoid heavy plotting/data size in js)
    cust_spending = sales_cust.groupby(['customer_id', 'customer_name']).agg(
        total_spending=('net_sales', 'sum'),
        loyalty_points=('loyalty_points', 'first')
    ).reset_index()
    loyalty_correlation_sample = cust_spending.head(200).to_dict(orient='records')
    
    # 8. Save Cleaned Files
    print(f"Saving CSV tables to {output_dir}...")
    sales_df.to_csv(os.path.join(output_dir, 'fact_sales_metrics.csv'), index=False)
    customer_df.to_csv(os.path.join(output_dir, 'dim_customer_metrics.csv'), index=False)
    dim_menu.to_csv(os.path.join(output_dir, 'dim_menu_metrics.csv'), index=False)
    location_df.to_csv(os.path.join(output_dir, 'dim_location_metrics.csv'), index=False)
    date_df.to_csv(os.path.join(output_dir, 'dim_date_metrics.csv'), index=False)
    
    print(f"Saving Master Excel sheet to {output_excel}...")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        sales_df.to_excel(writer, sheet_name='fact_sales_metrics', index=False)
        customer_df.to_excel(writer, sheet_name='dim_customer_metrics', index=False)
        dim_menu.to_excel(writer, sheet_name='dim_menu_metrics', index=False)
        location_df.to_excel(writer, sheet_name='dim_location_metrics', index=False)
        date_df.to_excel(writer, sheet_name='dim_date_metrics', index=False)
        
    # 9. Generate Static Visualizations (Matplotlib/Seaborn)
    print("Generating static charts...")
    sns.set_theme(style="darkgrid")
    
    # Chart 1: Monthly Sales & Profit Trend
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_stats['month_label'], monthly_stats['revenue'], marker='o', color='#3b82f6', label='Net Revenue', linewidth=2.5)
    plt.plot(monthly_stats['month_label'], monthly_stats['profit'], marker='s', color='#10b981', label='Operating Profit', linewidth=2.5)
    plt.title('Monthly Restaurant Performance Trends (2022–2024)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45)
    plt.xlabel('Year-Month')
    plt.ylabel('Amount (Rs.)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'monthly_sales_profit_trend.png'), dpi=300)
    plt.close()
    
    # Chart 2: Category Profit Margins
    plt.figure(figsize=(10, 6))
    sns.barplot(data=category_summary.sort_values(by='margin_pct', ascending=False), x='margin_pct', y='category_name', palette='crest', hue='category_name', legend=False)
    plt.title('Profit Margin Contribution by Food Category', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Profit Margin (%)')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'category_profit_margins.png'), dpi=300)
    plt.close()
    
    # Chart 3: BCG Matrix Scatter Plot
    plt.figure(figsize=(10, 7))
    bcg_colors = {'Star': '#10b981', 'Plowhorse': '#3b82f6', 'Puzzle': '#f59e0b', 'Dog': '#f43f5e'}
    sns.scatterplot(
        data=dim_menu,
        x='total_quantity_sold',
        y='unit_contribution_margin',
        hue='menu_category',
        palette=bcg_colors,
        s=100,
        edgecolor='black',
        alpha=0.85
    )
    plt.axvline(x=avg_popularity, color='gray', linestyle='--', linewidth=1.5, label='Popularity Benchmark')
    plt.axhline(y=avg_profitability, color='gray', linestyle='--', linewidth=1.5, label='Profitability Benchmark')
    plt.title('BCG Menu Engineering Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Total Quantity Sold')
    plt.ylabel('Unit Contribution Margin (Rs.)')
    plt.legend(title='BCG Classification')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'menu_bcg_matrix.png'), dpi=300)
    plt.close()
    
    # Chart 4: Customer Segment Profit Share
    plt.figure(figsize=(7, 7))
    plt.pie(
        customer_type_perf['profit'],
        labels=customer_type_perf['customer_type'],
        autopct='%1.1f%%',
        colors=['#6366f1', '#10b981', '#3b82f6', '#f59e0b'],
        startangle=140,
        textprops={'fontweight': 'bold', 'fontsize': 12}
    )
    plt.title('Profit Share Contribution by Customer Segment', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'customer_segments.png'), dpi=300)
    plt.close()
    
    # 10. Compile Interactive HTML Dashboard
    print("Compiling Interactive HTML Dashboard...")
    
    # Prepare data arrays for JSON
    global_kpis = {
        'total_revenue': float(sales_df['net_sales'].sum()),
        'total_profit': float(sales_df['profit'].sum()),
        'overall_margin': float(sales_df['profit'].sum() / sales_df['net_sales'].sum() * 100),
        'total_orders': int(sales_df['order_id'].nunique()),
        'aov': float(sales_df['net_sales'].sum() / sales_df['order_id'].nunique()),
        'avg_rating': float(sales_df['customer_rating'].mean()),
        'active_branches': int(location_df['location_id'].nunique()),
        'menu_items': int(menu_df['item_id'].nunique())
    }
    
    # Sort Menu items for tables
    menu_bcg_list = dim_menu[['item_name', 'category_name', 'total_quantity_sold', 'unit_contribution_margin', 'menu_category', 'base_price']].to_dict(orient='records')
    
    dashboard_data = {
        'kpis': global_kpis,
        'monthly_stats': monthly_stats[['month_label', 'revenue', 'profit']].to_dict(orient='records'),
        'category_stats': category_summary.to_dict(orient='records'),
        'branch_stats': branch_perf.to_dict(orient='records'),
        'menu_bcg': menu_bcg_list,
        'bcg_benchmarks': {
            'avg_qty': float(avg_popularity),
            'avg_margin': float(avg_profitability)
        },
        'day_perf': day_perf.to_dict(orient='records'),
        'order_type_perf': order_type_perf.to_dict(orient='records'),
        'payment_perf': payment_perf.to_dict(orient='records'),
        'customer_type_perf': customer_type_perf.to_dict(orient='records'),
        'age_group_perf': age_group_perf.to_dict(orient='records'),
        'loyalty_sample': loyalty_correlation_sample
    }
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Command Center KPI Dashboard</title>
    
    <!-- Outfit & Inter Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {{
            --bg-base: #090a0f;
            --bg-card: rgba(17, 19, 28, 0.75);
            --bg-card-hover: rgba(23, 26, 38, 0.9);
            --border-card: rgba(255, 255, 255, 0.05);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            
            --accent-blue: #3b82f6;
            --accent-teal: #0d9488;
            --accent-rose: #f43f5e;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-indigo: #6366f1;
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
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 35%),
                radial-gradient(at 100% 0%, rgba(13, 148, 136, 0.08) 0px, transparent 35%);
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
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
            background: rgba(17, 19, 28, 0.5);
            border-bottom: 1px solid var(--border-card);
            margin: -24px -24px 0 -24px;
            backdrop-filter: blur(8px);
            z-index: 10;
        }}
        
        header h1 {{
            font-size: 26px;
            background: linear-gradient(to right, #fff, #0d9488);
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
            background: var(--accent-teal);
            color: white;
            border-color: var(--accent-teal);
            box-shadow: 0 0 16px rgba(13, 148, 136, 0.4);
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
        .kpi-card.teal::before {{ background: var(--accent-teal); }}
        .kpi-card.emerald::before {{ background: var(--accent-emerald); }}
        .kpi-card.rose::before {{ background: var(--accent-rose); }}
        .kpi-card.indigo::before {{ background: var(--accent-indigo); }}
        .kpi-card.amber::before {{ background: var(--accent-amber); }}
        
        .kpi-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .kpi-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 26px;
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
        
        .badge.star {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        
        .badge.plowhorse {{
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        
        .badge.puzzle {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-amber);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        
        .badge.dog {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Restaurant command Center</h1>
            <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">Strategic & Operational Performance KPI Dashboard (₹ Currency Model)</p>
        </div>
        <div style="font-size: 14px; text-align: right; color: var(--text-secondary);">
            <div>Operating Branches: <span style="color: var(--accent-teal); font-weight: 600;">5 Active</span></div>
            <div style="font-size: 11px; margin-top: 4px;">Audit Baseline: Centralized Master DB</div>
        </div>
    </header>

    <div class="tab-navigation">
        <button class="tab-btn active" onclick="switchTab('finance')">Financial Command Center</button>
        <button class="tab-btn" onclick="switchTab('menu')">Menu Engineering & BCG</button>
        <button class="tab-btn" onclick="switchTab('operations')">Operations & Channels</button>
        <button class="tab-btn" onclick="switchTab('customer')">Customer Dimensions</button>
    </div>

    <!-- VIEW 1: FINANCIAL COMMAND CENTER -->
    <div id="view-finance" class="dashboard-view active">
        <!-- KPIs -->
        <div class="grid-6">
            <div class="glass-panel kpi-card teal">
                <div class="kpi-title">Net Revenue</div>
                <div class="kpi-value" id="kpi-rev">₹0</div>
            </div>
            <div class="glass-panel kpi-card emerald">
                <div class="kpi-title">Operating Profit</div>
                <div class="kpi-value" id="kpi-profit">₹0</div>
            </div>
            <div class="glass-panel kpi-card teal">
                <div class="kpi-title">Profit Margin %</div>
                <div class="kpi-value" id="kpi-margin">0.0%</div>
            </div>
            <div class="glass-panel kpi-card indigo">
                <div class="kpi-title">Total Orders</div>
                <div class="kpi-value" id="kpi-orders">0</div>
            </div>
            <div class="glass-panel kpi-card blue">
                <div class="kpi-title">Avg Order Value</div>
                <div class="kpi-value" id="kpi-aov">₹0</div>
            </div>
            <div class="glass-panel kpi-card amber">
                <div class="kpi-title">Avg Rating</div>
                <div class="kpi-value" id="kpi-rating">0.0 ⭐</div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Monthly Trends -->
            <div class="glass-panel">
                <h3>Monthly Revenue & Profit Trends (MoM)</h3>
                <div class="chart-container">
                    <canvas id="chart-monthly-trend"></canvas>
                </div>
            </div>
            
            <!-- Category Breakdown -->
            <div class="glass-panel">
                <h3>Food Category Revenue vs. Margin Contribution</h3>
                <div class="chart-container">
                    <canvas id="chart-category-breakdown"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Branch Rankings -->
        <div class="glass-panel">
            <h3>Branch Performance Rankings & Seating Utilization</h3>
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

    <!-- VIEW 2: MENU ENGINEERING & BCG -->
    <div id="view-menu" class="dashboard-view">
        <div class="grid-2">
            <!-- BCG Scatter -->
            <div class="glass-panel">
                <h3>BCG Matrix Scatter Plot</h3>
                <p style="color: var(--text-secondary); font-size: 11px; margin-top: 4px;">Menu items mapped by popularity (total sold) vs profitability (unit contribution margin).</p>
                <div class="chart-container" style="height: 350px;">
                    <canvas id="chart-bcg-scatter"></canvas>
                </div>
            </div>
            
            <!-- BCG Matrix Summary & Breakdown -->
            <div class="glass-panel">
                <h3>Menu Engineering Classifications</h3>
                <div style="display: flex; gap: 10px; margin-top: 15px; margin-bottom: 15px;">
                    <div style="padding: 10px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; flex: 1; text-align: center;">
                        <div style="font-size: 11px; color: var(--accent-emerald); font-weight:600; text-transform: uppercase;">Avg Qty Benchmark</div>
                        <div style="font-size: 20px; font-weight:800; font-family:'Outfit';" id="bcg-bench-qty">0</div>
                    </div>
                    <div style="padding: 10px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; flex: 1; text-align: center;">
                        <div style="font-size: 11px; color: var(--accent-blue); font-weight:600; text-transform: uppercase;">Avg Margin Benchmark</div>
                        <div style="font-size: 20px; font-weight:800; font-family:'Outfit';" id="bcg-bench-margin">₹0</div>
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

    <!-- VIEW 3: OPERATIONS & CHANNELS -->
    <div id="view-operations" class="dashboard-view">
        <div class="grid-3">
            <!-- Order Type -->
            <div class="glass-panel">
                <h3>Order Type Mix (Share %)</h3>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="chart-order-type"></canvas>
                </div>
            </div>
            
            <!-- Payment Method -->
            <div class="glass-panel">
                <h3>Payment Method Mix (Share %)</h3>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="chart-payment-method"></canvas>
                </div>
            </div>
            
            <!-- Day of Week -->
            <div class="glass-panel">
                <h3>Weekly Order Pattern</h3>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="chart-weekly-pattern"></canvas>
                </div>
            </div>
        </div>
        
        <div class="glass-panel">
            <h3>Operational Efficiency Guidelines</h3>
            <ul style="margin-left: 18px; font-size: 13px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 8px; line-height: 1.4; margin-top: 10px;">
                <li><strong>Seat Optimization</strong>: Low profit-per-seat branches (such as Suburb) should consider restructuring seat layout or promoting dine-in combo offers during off-peak hours.</li>
                <li><strong>Channel Shifts</strong>: High delivery volume represents low physical space usage but adds packaging cost. Dine-in orders have higher customer ratings and should be encouraged via loyalty promotions.</li>
            </ul>
        </div>
    </div>

    <!-- VIEW 4: CUSTOMER DIMENSIONS -->
    <div id="view-customer" class="dashboard-view">
        <div class="grid-2">
            <!-- Customer Segment share -->
            <div class="glass-panel">
                <h3>Customer Segment Profit Contribution</h3>
                <div class="chart-container">
                    <canvas id="chart-customer-segments"></canvas>
                </div>
            </div>
            
            <!-- Age Group spending -->
            <div class="glass-panel">
                <h3>Age Group Spending Profile (Net Revenue)</h3>
                <div class="chart-container">
                    <canvas id="chart-age-spending"></canvas>
                </div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Loyalty Spending correlation -->
            <div class="glass-panel">
                <h3>Loyalty Points vs. Customer Spending</h3>
                <p style="color: var(--text-secondary); font-size: 11px; margin-top: 4px;">Scatter plot showing correlation between customer loyalty points and lifetime spending (Rs.).</p>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="chart-loyalty-scatter"></canvas>
                </div>
            </div>
            
            <!-- Customer segment stats -->
            <div class="glass-panel">
                <h3>Customer Segment Value Metrics</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Customer Type</th>
                                <th>Revenue Contribution</th>
                                <th>Profit Contribution</th>
                                <th>Unique Customer Count</th>
                            </tr>
                        </thead>
                        <tbody id="customer-table-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dashboardData = {json.dumps(dashboard_data, indent=4)};
        
        // Tab switching
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.dashboard-view').forEach(view => view.classList.remove('active'));
            
            let activeBtn = document.querySelector(`.tab-btn[onclick="switchTab('${{tabId}}')"]`);
            if (activeBtn) activeBtn.classList.add('active');
            
            let targetView = document.getElementById(`view-${{tabId}}`);
            if (targetView) targetView.classList.add('active');
        }}

        // Formatters
        const formatINR = (val) => {{
            return new Intl.NumberFormat('en-IN', {{ style: 'currency', currency: 'INR', maximumFractionDigits: 0 }}).format(val);
        }};
        
        const formatInt = (val) => {{
            return new Intl.NumberFormat('en-IN').format(val);
        }};

        // Set KPIs
        document.getElementById('kpi-rev').innerText = formatINR(dashboardData.kpis.total_revenue);
        document.getElementById('kpi-profit').innerText = formatINR(dashboardData.kpis.total_profit);
        document.getElementById('kpi-margin').innerText = dashboardData.kpis.overall_margin.toFixed(2) + "%";
        document.getElementById('kpi-orders').innerText = formatInt(dashboardData.kpis.total_orders);
        document.getElementById('kpi-aov').innerText = formatINR(dashboardData.kpis.aov);
        document.getElementById('kpi-rating').innerText = dashboardData.kpis.avg_rating.toFixed(2) + " ⭐";
        
        document.getElementById('bcg-bench-qty').innerText = formatInt(dashboardData.bcg_benchmarks.avg_qty);
        document.getElementById('bcg-bench-margin').innerText = formatINR(dashboardData.bcg_benchmarks.avg_margin);

        // Populate Branch Table
        const branchTbody = document.getElementById('branch-table-tbody');
        dashboardData.branch_stats.forEach(b => {{
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
        dashboardData.menu_bcg.forEach(item => {{
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

        // Populate Customer Table
        const customerTbody = document.getElementById('customer-table-tbody');
        dashboardData.customer_type_perf.forEach(c => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600;">${{c.customer_type}}</td>
                <td>${{formatINR(c.revenue)}}</td>
                <td style="font-weight:600; color: var(--accent-emerald);">${{formatINR(c.profit)}}</td>
                <td>${{formatInt(c.customers)}}</td>
            `;
            customerTbody.appendChild(tr);
        }});

        // ----------------------------------------------------
        // CHARTS SETUP (Chart.js)
        // ----------------------------------------------------
        
        // 1. Monthly Trends Chart
        const ctxMonthly = document.getElementById('chart-monthly-trend').getContext('2d');
        const months = dashboardData.monthly_stats.map(m => m.month_label);
        const revenues = dashboardData.monthly_stats.map(m => m.revenue);
        const profits = dashboardData.monthly_stats.map(m => m.profit);
        
        new Chart(ctxMonthly, {{
            type: 'line',
            data: {{
                labels: months,
                datasets: [
                    {{
                        label: 'Net Revenue',
                        data: revenues,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2.5,
                        pointRadius: 3,
                        fill: true
                    }},
                    {{
                        label: 'Operating Profit',
                        data: profits,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2.5,
                        pointRadius: 3,
                        fill: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#f3f4f6' }} }}
                }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#9ca3af' }} }},
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }} }}
                }}
            }}
        }});

        // 2. Category Breakdown Chart
        const ctxCategory = document.getElementById('chart-category-breakdown').getContext('2d');
        const categories = dashboardData.category_stats.map(c => c.category_name);
        const catRevenues = dashboardData.category_stats.map(c => c.revenue);
        const catMargins = dashboardData.category_stats.map(c => c.margin_pct);
        
        new Chart(ctxCategory, {{
            type: 'bar',
            data: {{
                labels: categories,
                datasets: [
                    {{
                        label: 'Revenue (Rs.)',
                        data: catRevenues,
                        backgroundColor: 'rgba(99, 102, 241, 0.8)',
                        borderColor: '#6366f1',
                        borderWidth: 1.5,
                        borderRadius: 6,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Margin %',
                        data: catMargins,
                        type: 'line',
                        borderColor: '#eab308',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#eab308',
                        fill: false,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#f3f4f6' }} }}
                }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#9ca3af' }} }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                        ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#9ca3af', callback: function(value) {{ return value + '%'; }} }}
                    }}
                }}
            }}
        }});

        // 3. BCG Scatter Plot
        const ctxBCG = document.getElementById('chart-bcg-scatter').getContext('2d');
        const bcgPoints = dashboardData.menu_bcg.map(item => ({{
            x: item.total_quantity_sold,
            y: item.unit_contribution_margin,
            label: item.item_name,
            segment: item.menu_category
        }}));
        
        const segmentColors = {{
            'Star': '#10b981',
            'Plowhorse': '#3b82f6',
            'Puzzle': '#f59e0b',
            'Dog': '#f43f5e'
        }};
        
        new Chart(ctxBCG, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    data: bcgPoints,
                    pointBackgroundColor: bcgPoints.map(p => segmentColors[p.segment]),
                    pointBorderColor: 'rgba(0, 0, 0, 0.6)',
                    pointRadius: 8,
                    pointHoverRadius: 10
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
                    x: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                        ticks: {{ color: '#9ca3af' }},
                        title: {{ display: true, text: 'Total Quantity Sold', color: '#9ca3af' }}
                    }},
                    y: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                        ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }},
                        title: {{ display: true, text: 'Unit Contribution Margin (Rs.)', color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        // 4. Order Type mix
        const ctxOrderType = document.getElementById('chart-order-type').getContext('2d');
        new Chart(ctxOrderType, {{
            type: 'doughnut',
            data: {{
                labels: dashboardData.order_type_perf.map(o => o.order_type),
                datasets: [{{
                    data: dashboardData.order_type_perf.map(o => o.net_sales),
                    backgroundColor: ['#6366f1', '#10b981', '#3b82f6'],
                    borderWidth: 1,
                    borderColor: 'rgba(0,0,0,0.5)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: '#f3f4f6' }} }}
                }}
            }}
        }});

        // 5. Payment Method mix
        const ctxPayment = document.getElementById('chart-payment-method').getContext('2d');
        new Chart(ctxPayment, {{
            type: 'doughnut',
            data: {{
                labels: dashboardData.payment_perf.map(p => p.payment_method),
                datasets: [{{
                    data: dashboardData.payment_perf.map(p => p.net_sales),
                    backgroundColor: ['#0d9488', '#f59e0b', '#3b82f6', '#6366f1', '#f43f5e'],
                    borderWidth: 1,
                    borderColor: 'rgba(0,0,0,0.5)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: '#f3f4f6', boxWidth: 12 }} }}
                }}
            }}
        }});

        // 6. Weekly Pattern
        const ctxWeekly = document.getElementById('chart-weekly-pattern').getContext('2d');
        new Chart(ctxWeekly, {{
            type: 'bar',
            data: {{
                labels: dashboardData.day_perf.map(d => d.day_name),
                datasets: [{{
                    data: dashboardData.day_perf.map(d => d.orders),
                    backgroundColor: 'rgba(13, 148, 136, 0.8)',
                    borderColor: '#0d9488',
                    borderWidth: 1,
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

        // 7. Customer segments
        const ctxCustType = document.getElementById('chart-customer-segments').getContext('2d');
        new Chart(ctxCustType, {{
            type: 'pie',
            data: {{
                labels: dashboardData.customer_type_perf.map(c => c.customer_type),
                datasets: [{{
                    data: dashboardData.customer_type_perf.map(c => c.profit),
                    backgroundColor: ['#6366f1', '#10b981', '#3b82f6', '#f59e0b'],
                    borderWidth: 1,
                    borderColor: 'rgba(0,0,0,0.5)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#f3f4f6' }} }}
                }}
            }}
        }});

        // 8. Age Spending
        const ctxAge = document.getElementById('chart-age-spending').getContext('2d');
        new Chart(ctxAge, {{
            type: 'bar',
            data: {{
                labels: dashboardData.age_group_perf.map(a => a.age_group),
                datasets: [{{
                    label: 'Net Revenue',
                    data: dashboardData.age_group_perf.map(a => a.revenue),
                    backgroundColor: 'rgba(59, 130, 246, 0.8)',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ display: false }} }},
                    y: {{ ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }}, grid: {{ color: 'rgba(255, 255, 255, 0.04)' }} }}
                }}
            }}
        }});

        // 9. Loyalty spending scatter
        const ctxLoyalty = document.getElementById('chart-loyalty-scatter').getContext('2d');
        const loyaltyPoints = dashboardData.loyalty_sample.map(c => ({{
            x: c.loyalty_points,
            y: c.total_spending,
            label: c.customer_name
        }}));
        
        new Chart(ctxLoyalty, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    data: loyaltyPoints,
                    pointBackgroundColor: 'rgba(99, 102, 241, 0.7)',
                    pointBorderColor: '#6366f1',
                    pointRadius: 4
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
                                return `${{p.label}}: Points: ${{formatInt(p.x)}} | Spend: ${{formatINR(p.y)}}`;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#9ca3af' }}, title: {{ display: true, text: 'Customer Loyalty Points', color: '#9ca3af' }} }},
                    y: {{ ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }}, title: {{ display: true, text: 'Total Spending (Rs.)', color: '#9ca3af' }} }}
                }}
            }}
        }});
        
    </script>
</body>
</html>
"""
    
    print(f"Writing interactive HTML dashboard to {dashboard_file}...")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    print("==================================================================")
    print("KEY KPI COMMAND CENTER PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================================")

if __name__ == "__main__":
    run_command_center()
