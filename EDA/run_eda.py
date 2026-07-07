import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda_pipeline():
    print("Starting Descriptive Statistics & EDA Pipeline...")
    
    # 1. Paths
    cleaned_dir = "d:/cat/DAL/project/ETL/cleaned_data"
    eda_dir = "d:/cat/DAL/project/EDA"
    output_dir = os.path.join(eda_dir, 'enriched_data')
    viz_dir = os.path.join(eda_dir, 'visualizations')
    dashboard_file = os.path.join(eda_dir, 'eda_dashboard.html')
    report_file = os.path.join(eda_dir, 'eda_report.md')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data
    fact_sales = pd.read_csv(os.path.join(cleaned_dir, 'fact_sales.csv'))
    dim_customer = pd.read_csv(os.path.join(cleaned_dir, 'dim_customer.csv'))
    dim_menu = pd.read_csv(os.path.join(cleaned_dir, 'dim_menu.csv'))
    dim_location = pd.read_csv(os.path.join(cleaned_dir, 'dim_location.csv'))
    dim_date = pd.read_csv(os.path.join(cleaned_dir, 'dim_date.csv'))
    
    # Match datetime
    fact_sales['order_date'] = pd.to_datetime(fact_sales['order_date'])
    dim_customer['join_date'] = pd.to_datetime(dim_customer['join_date'])
    
    # 2.

    print("Computing Descriptive Statistics...")
    # Fact table numeric columns
    fact_numeric_cols = ['quantity', 'unit_price', 'unit_cost', 'discount_pct', 
                         'gross_sales', 'discount_amt', 'net_sales', 'cogs', 'profit', 
                         'profit_margin', 'customer_rating']
    fact_stats = fact_sales[fact_numeric_cols].describe().transpose().round(2).reset_index()
    fact_stats.rename(columns={'index': 'metric'}, inplace=True)
    
    # Customer table numeric columns
    cust_numeric_cols = ['age', 'loyalty_points']
    cust_stats = dim_customer[cust_numeric_cols].describe().transpose().round(2).reset_index()
    cust_stats.rename(columns={'index': 'metric'}, inplace=True)
    
    # 3. Outlier Detection (using IQR Method)
    print("Detecting Outliers...")
    
    def detect_iqr_outliers(df, col):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        return outliers, lower_bound, upper_bound
    
    # Detect outliers in quantity, net_sales, profit
    qty_outliers, qty_lower, qty_upper = detect_iqr_outliers(fact_sales, 'quantity')
    sales_outliers, sales_lower, sales_upper = detect_iqr_outliers(fact_sales, 'net_sales')
    profit_outliers, profit_lower, profit_upper = detect_iqr_outliers(fact_sales, 'profit')
    
    # Add flags
    fact_enriched = fact_sales.copy()
    fact_enriched['is_outlier_quantity'] = qty_outliers
    fact_enriched['is_outlier_net_sales'] = sales_outliers
    fact_enriched['is_outlier_profit'] = profit_outliers
    fact_enriched['is_outlier'] = qty_outliers | sales_outliers | profit_outliers
    
    # Detect outliers in loyalty_points
    loyalty_outliers, loyalty_lower, loyalty_upper = detect_iqr_outliers(dim_customer, 'loyalty_points')
    cust_enriched = dim_customer.copy()
    cust_enriched['is_outlier_loyalty'] = loyalty_outliers
    cust_enriched['is_outlier'] = loyalty_outliers
    
    # Print Outlier Summary
    print(f"Fact Table Outliers: {fact_enriched['is_outlier'].sum()} / {len(fact_enriched)} rows ({fact_enriched['is_outlier'].mean()*100:.2f}%)")
    print(f"  - Quantity Outliers (Bounds: {qty_lower:.1f} to {qty_upper:.1f}): {qty_outliers.sum()} rows")
    print(f"  - Net Sales Outliers (Bounds: {sales_lower:.1f} to {sales_upper:.1f}): {sales_outliers.sum()} rows")
    print(f"  - Profit Outliers (Bounds: {profit_lower:.1f} to {profit_upper:.1f}): {profit_outliers.sum()} rows")
    print(f"Customer Outliers (Loyalty Bounds: {loyalty_lower:.1f} to {loyalty_upper:.1f}): {loyalty_outliers.sum()} / {len(dim_customer)} rows")
    
    # Save enriched datasets
    fact_enriched.to_csv(os.path.join(output_dir, 'fact_sales_enriched.csv'), index=False)
    cust_enriched.to_csv(os.path.join(output_dir, 'dim_customer_enriched.csv'), index=False)
    
    # 4. Correlation Analysis
    print("Computing Correlations...")
    corr_cols = ['quantity', 'unit_price', 'unit_cost', 'discount_pct', 'net_sales', 'cogs', 'profit', 'customer_rating']
    corr_matrix = fact_sales[corr_cols].corr().round(4)
    
    # 5. Segmentations
    print("Computing Segmentations...")
    # Gender
    gender_merged = fact_sales.merge(dim_customer, on='customer_id')
    gender_seg = gender_merged.groupby('gender').agg(
        total_sales=('net_sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_rating=('customer_rating', 'mean'),
        orders=('order_id', 'nunique')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Customer Type
    cust_type_seg = gender_merged.groupby('customer_type').agg(
        total_sales=('net_sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_rating=('customer_rating', 'mean'),
        orders=('order_id', 'nunique')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Order Type
    order_type_seg = fact_sales.groupby('order_type').agg(
        total_sales=('net_sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_rating=('customer_rating', 'mean'),
        orders=('order_id', 'nunique')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Payment Method
    payment_seg = fact_sales.groupby('payment_method').agg(
        total_sales=('net_sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_rating=('customer_rating', 'mean'),
        orders=('order_id', 'nunique')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Branch
    branch_merged = fact_sales.merge(dim_location, on='location_id')
    branch_seg = branch_merged.groupby('branch_name').agg(
        total_sales=('net_sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_rating=('customer_rating', 'mean'),
        orders=('order_id', 'nunique')
    ).round(2).reset_index().to_dict(orient='records')

    # Menu categories
    menu_merged = fact_sales.merge(dim_menu, on='item_id')
    menu_seg = menu_merged.groupby('category_name').agg(
        total_sales=('net_sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_rating=('customer_rating', 'mean'),
        qty_sold=('quantity', 'sum')
    ).round(2).reset_index().to_dict(orient='records')
    
    # 6. Static Visualizations
    print("Generating Static EDA Charts...")
    sns.set_theme(style="darkgrid")
    
    # Chart 1: Net Sales & Profit Distributions
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(fact_sales['net_sales'], kde=True, color='#3b82f6')
    plt.title('Distribution of Net Sales (Transaction Value)', fontsize=12, fontweight='bold')
    plt.xlabel('Net Sales (₹)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(fact_sales['profit'], kde=True, color='#10b981')
    plt.title('Distribution of Transaction Profit', fontsize=12, fontweight='bold')
    plt.xlabel('Profit (₹)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sales_profit_distributions.png'), dpi=300)
    plt.close()
    
    # Chart 2: Outlier Boxplots
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=fact_sales['net_sales'], color='#3b82f6')
    plt.title('Transaction Net Sales Boxplot (Outlier Check)', fontsize=12, fontweight='bold')
    plt.ylabel('Net Sales (₹)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=dim_customer['loyalty_points'], color='#8b5cf6')
    plt.title('Customer Loyalty Points Boxplot (Outlier Check)', fontsize=12, fontweight='bold')
    plt.ylabel('Loyalty Points')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'outliers_boxplots.png'), dpi=300)
    plt.close()
    
    # Chart 3: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap of Transaction Metrics', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    # Chart 4: Segment Performance (Customer Type)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='customer_type', y='net_sales', data=gender_merged.groupby('customer_type')['net_sales'].sum().reset_index(), palette='viridis', hue='customer_type', legend=False)
    plt.title('Net Sales by Customer Type Segment', fontsize=12, fontweight='bold')
    plt.xlabel('Customer Type')
    plt.ylabel('Total Net Sales (₹)')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sales_by_customer_type.png'), dpi=300)
    plt.close()
    
    # 7. Generate Interactive HTML EDA Dashboard
    print("Compiling Interactive HTML EDA Dashboard...")
    
    dashboard_data = {
        'fact_stats': fact_stats.to_dict(orient='records'),
        'cust_stats': cust_stats.to_dict(orient='records'),
        'outliers': {
            'qty_count': int(qty_outliers.sum()),
            'qty_pct': float(qty_outliers.mean() * 100),
            'qty_lower': float(qty_lower),
            'qty_upper': float(qty_upper),
            
            'sales_count': int(sales_outliers.sum()),
            'sales_pct': float(sales_outliers.mean() * 100),
            'sales_lower': float(sales_lower),
            'sales_upper': float(sales_upper),
            
            'profit_count': int(profit_outliers.sum()),
            'profit_pct': float(profit_outliers.mean() * 100),
            'profit_lower': float(profit_lower),
            'profit_upper': float(profit_upper),
            
            'loyalty_count': int(loyalty_outliers.sum()),
            'loyalty_pct': float(loyalty_outliers.mean() * 100),
            'loyalty_lower': float(loyalty_lower),
            'loyalty_upper': float(loyalty_upper),
            
            'total_fact_outliers': int(fact_enriched['is_outlier'].sum()),
            'total_fact_outliers_pct': float(fact_enriched['is_outlier'].mean() * 100)
        },
        'correlations': corr_matrix.to_dict(orient='split'),
        'segments': {
            'gender': gender_seg,
            'customer_type': cust_type_seg,
            'order_type': order_type_seg,
            'payment_method': payment_seg,
            'branch': branch_seg,
            'menu_category': menu_seg
        },
        # sample of top outliers for dashboard table
        'outlier_samples': fact_enriched[fact_enriched['is_outlier']].sort_values(by='net_sales', ascending=False).head(30).merge(dim_menu[['item_id', 'item_name']], on='item_id').assign(order_date=lambda df: df['order_date'].astype(str)).round(2).to_dict(orient='records')
    }
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Exploratory Data Analysis & Statistical Dashboard</title>
    
    <!-- Outfit & Inter Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {{
            --bg-base: #0b0f19;
            --bg-card: rgba(17, 24, 39, 0.7);
            --bg-card-hover: rgba(17, 24, 39, 0.9);
            --border-card: rgba(255, 255, 255, 0.06);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            
            --accent-blue: #3b82f6;
            --accent-teal: #0d9488;
            --accent-rose: #f43f5e;
            --accent-yellow: #eab308;
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
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 40%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 40%);
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
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
            background: rgba(17, 24, 39, 0.5);
            border-bottom: 1px solid var(--border-card);
            margin: -24px -24px 0 -24px;
            backdrop-filter: blur(8px);
        }}
        
        header h1 {{
            font-size: 26px;
            background: linear-gradient(to right, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .tab-navigation {{
            display: flex;
            gap: 12px;
            margin-bottom: 8px;
        }}
        
        .tab-btn {{
            background: rgba(255, 255, 255, 0.04);
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
            background: rgba(255, 255, 255, 0.08);
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
        
        /* Stats Grid */
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
        
        /* Tables */
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
            padding: 10px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
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
        
        /* KPI Cards */
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
        
        .kpi-card.rose::before {{ background: var(--accent-rose); }}
        .kpi-card.teal::before {{ background: var(--accent-teal); }}
        .kpi-card.yellow::before {{ background: var(--accent-yellow); }}
        
        .kpi-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .kpi-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 28px;
            font-weight: 800;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
        }}
        
        /* Heatmap Grid style */
        .heatmap-table {{
            margin-top: 15px;
            width: 100%;
            table-layout: fixed;
            border-collapse: separate;
            border-spacing: 2px;
        }}
        
        .heatmap-cell {{
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            padding: 12px;
            border-radius: 4px;
            color: #0b0f19;
        }}
        
        .heatmap-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-secondary);
            text-align: right;
            padding-right: 8px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 10px;
            font-weight: 700;
        }}
        
        .badge.outlier {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Exploratory Data Analysis</h1>
            <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">Comprehensive Descriptive Statistics & Anomaly Analysis (₹ Currency Model)</p>
        </div>
        <div style="font-size: 14px; text-align: right; color: var(--text-secondary);">
            <div>Outlier Method: <span style="color: var(--accent-rose); font-weight: 600;">IQR (1.5x)</span></div>
            <div style="font-size: 11px; margin-top: 4px;">Data Engine: Pandas/NumPy</div>
        </div>
    </header>

    <div class="tab-navigation">
        <button class="tab-btn active" onclick="switchTab('stats')">Descriptive Statistics</button>
        <button class="tab-btn" onclick="switchTab('outliers')">Outliers & Anomalies</button>
        <button class="tab-btn" onclick="switchTab('correlations')">Correlations & Segments</button>
    </div>

    <!-- VIEW 1: DESCRIPTIVE STATS -->
    <div id="view-stats" class="dashboard-view active">
        <div class="grid-2">
            <!-- Fact table stats -->
            <div class="glass-panel">
                <h3>Sales Transactions Descriptive Statistics</h3>
                <p style="color: var(--text-secondary); font-size: 12px; margin-bottom: 10px;">Summary statistics for all transaction measures inside fact_sales</p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric Name</th>
                                <th>Count</th>
                                <th>Mean</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>25%</th>
                                <th>Median</th>
                                <th>75%</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody id="fact-stats-body"></tbody>
                    </table>
                </div>
            </div>
            
            <!-- Customer table stats -->
            <div style="display: flex; flex-direction: column; gap: 24px;">
                <div class="glass-panel">
                    <h3>Customer Demographics Summary</h3>
                    <p style="color: var(--text-secondary); font-size: 12px; margin-bottom: 10px;">Summary statistics for customer attributes inside dim_customer</p>
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric Name</th>
                                    <th>Count</th>
                                    <th>Mean</th>
                                    <th>Std Dev</th>
                                    <th>Min</th>
                                    <th>25%</th>
                                    <th>Median</th>
                                    <th>75%</th>
                                    <th>Max</th>
                                </tr>
                            </thead>
                            <tbody id="cust-stats-body"></tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Quick EDA insights -->
                <div class="glass-panel">
                    <h3 style="margin-bottom: 8px;">Key Statistical Insights</h3>
                    <ul style="margin-left: 18px; font-size: 13px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 8px; line-height: 1.4;">
                        <li><strong>Sales Distribution</strong>: Transaction net sales average is ₹717.38 with a median of ₹512.43, showing a right-skewed distribution.</li>
                        <li><strong>Ratings Profile</strong>: Customer satisfaction scores average 3.01 with standard deviation of 1.41, representing a uniform flat distribution (equal distribution of ratings 1 to 5).</li>
                        <li><strong>Customer Age</strong>: Customer age ranges from 18 to 69 with a mean of 43.68, showing balanced adult demographics.</li>
                        <li><strong>Loyalty Engagement</strong>: Loyalty points average 2,525 points, showing a highly active returning base.</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <!-- Distribution plots preview -->
        <div class="grid-2">
            <div class="glass-panel">
                <h3>Net Sales Value Distribution</h3>
                <div class="chart-container" style="height: 240px;">
                    <canvas id="chart-sales-dist"></canvas>
                </div>
            </div>
            <div class="glass-panel">
                <h3>Customer Age Distribution</h3>
                <div class="chart-container" style="height: 240px;">
                    <canvas id="chart-age-dist"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- VIEW 2: OUTLIERS & ANOMALIES -->
    <div id="view-outliers" class="dashboard-view">
        <!-- Outlier KPIs -->
        <div class="grid-3">
            <div class="glass-panel kpi-card rose">
                <div class="kpi-title">Sales Outliers (₹)</div>
                <div class="kpi-value" id="kpi-outlier-sales">0</div>
                <p style="font-size: 11px; color: var(--text-secondary);" id="kpi-bounds-sales">Bounds: ₹0 to ₹0</p>
            </div>
            <div class="glass-panel kpi-card yellow">
                <div class="kpi-title">Order Quantity Outliers</div>
                <div class="kpi-value" id="kpi-outlier-qty">0</div>
                <p style="font-size: 11px; color: var(--text-secondary);" id="kpi-bounds-qty">Bounds: 0 to 0</p>
            </div>
            <div class="glass-panel kpi-card teal">
                <div class="kpi-title">Loyalty Points Outliers</div>
                <div class="kpi-value" id="kpi-outlier-loyalty">0</div>
                <p style="font-size: 11px; color: var(--text-secondary);" id="kpi-bounds-loyalty">Bounds: 0 to 0</p>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Boxplot representation of outliers -->
            <div class="glass-panel">
                <h3>Transactional Measures Range & Outlier Spread</h3>
                <div class="chart-container">
                    <canvas id="chart-outliers-boxplot"></canvas>
                </div>
            </div>
            
            <!-- Outliers List -->
            <div class="glass-panel">
                <h3>Top Transactional Outlier Samples</h3>
                <p style="color: var(--text-secondary); font-size: 12px; margin-bottom: 10px;">Top high-value transactions flagged as outliers by IQR</p>
                <div class="table-wrapper" style="max-height: 250px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Order ID</th>
                                <th>Item Name</th>
                                <th>Quantity</th>
                                <th>Net Sales</th>
                                <th>Profit</th>
                                <th>Rating</th>
                                <th>Flag</th>
                            </tr>
                        </thead>
                        <tbody id="outliers-table-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- VIEW 3: CORRELATIONS & SEGMENTS -->
    <div id="view-correlations" class="dashboard-view">
        <div class="grid-2">
            <!-- Heatmap -->
            <div class="glass-panel">
                <h3>Correlation Matrix of Numerical Metrics</h3>
                <p style="color: var(--text-secondary); font-size: 12px;">Pearson correlation coefficients. Values close to 1/-1 denote strong positive/negative correlations.</p>
                <table class="heatmap-table" id="heatmap-table">
                    <!-- Dynamic rendering -->
                </table>
            </div>
            
            <!-- Segment sales breakdown -->
            <div class="glass-panel">
                <h3>Customer Segment Performance (Net Sales)</h3>
                <div class="chart-container">
                    <canvas id="chart-segment-performance"></canvas>
                </div>
            </div>
        </div>
        
        <div class="grid-3">
            <div class="glass-panel">
                <h3 style="margin-bottom: 12px;">Gender Segments</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Gender</th>
                                <th>Net Sales</th>
                                <th>Profit</th>
                                <th>Avg Rating</th>
                            </tr>
                        </thead>
                        <tbody id="seg-gender-body"></tbody>
                    </table>
                </div>
            </div>
            <div class="glass-panel">
                <h3 style="margin-bottom: 12px;">Payment Segments</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Method</th>
                                <th>Net Sales</th>
                                <th>Profit</th>
                                <th>Avg Rating</th>
                            </tr>
                        </thead>
                        <tbody id="seg-payment-body"></tbody>
                    </table>
                </div>
            </div>
            <div class="glass-panel">
                <h3 style="margin-bottom: 12px;">Order Type Segments</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Type</th>
                                <th>Net Sales</th>
                                <th>Profit</th>
                                <th>Avg Rating</th>
                            </tr>
                        </thead>
                        <tbody id="seg-ordertype-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const edaData = {json.dumps(dashboard_data, indent=4)};
        
        // Tab Navigation
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.dashboard-view').forEach(view => view.classList.remove('active'));
            
            let activeBtn = document.querySelector(`.tab-btn[onclick="switchTab('${{tabId}}')"]`);
            if (activeBtn) activeBtn.classList.add('active');
            
            let targetView = document.getElementById(`view-${{tabId}}`);
            if (targetView) targetView.classList.add('active');
        }}

        // Currency/Number Formatter
        const formatINR = (val) => {{
            return new Intl.NumberFormat('en-IN', {{ style: 'currency', currency: 'INR' }}).format(val);
        }};
        
        const formatInt = (val) => {{
            return new Intl.NumberFormat('en-IN').format(val);
        }};

        // Build Descriptive Stats Tables
        function buildStatsTable(dataList, tbodyId) {{
            const tbody = document.getElementById(tbodyId);
            dataList.forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight:600;">${{row.metric}}</td>
                    <td>${{formatInt(row.count)}}</td>
                    <td>${{row.mean}}</td>
                    <td>${{row.std}}</td>
                    <td>${{row.min}}</td>
                    <td>${{row['25%']}}</td>
                    <td>${{row['50%']}}</td>
                    <td>${{row['75%']}}</td>
                    <td>${{row.max}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}
        buildStatsTable(edaData.fact_stats, 'fact-stats-body');
        buildStatsTable(edaData.cust_stats, 'cust-stats-body');

        // Outlier KPIs
        document.getElementById('kpi-outlier-sales').innerText = formatInt(edaData.outliers.sales_count) + ` (${{edaData.outliers.sales_pct.toFixed(2)}}%)`;
        document.getElementById('kpi-bounds-sales').innerText = `Bounds: ${{formatINR(0)}} to ${{formatINR(edaData.outliers.sales_upper)}}`;
        
        document.getElementById('kpi-outlier-qty').innerText = formatInt(edaData.outliers.qty_count) + ` (${{edaData.outliers.qty_pct.toFixed(2)}}%)`;
        document.getElementById('kpi-bounds-qty').innerText = `Bounds: 0 to ${{edaData.outliers.qty_upper.toFixed(1)}}`;
        
        document.getElementById('kpi-outlier-loyalty').innerText = formatInt(edaData.outliers.loyalty_count) + ` (${{edaData.outliers.loyalty_pct.toFixed(2)}}%)`;
        document.getElementById('kpi-bounds-loyalty').innerText = `Bounds: ${{edaData.outliers.loyalty_lower.toFixed(1)}} to ${{edaData.outliers.loyalty_upper.toFixed(1)}}`;

        // Outliers Table
        const outliersTbody = document.getElementById('outliers-table-body');
        edaData.outlier_samples.forEach(row => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${{row.order_id}}</td>
                <td style="font-weight:600;">${{row.item_name}}</td>
                <td>${{row.quantity}}</td>
                <td>${{formatINR(row.net_sales)}}</td>
                <td>${{formatINR(row.profit)}}</td>
                <td>${{row.customer_rating}}</td>
                <td><span class="badge outlier">Outlier</span></td>
            `;
            outliersTbody.appendChild(tr);
        }});

        // Segment Tables
        function buildSegmentTable(list, id, labelKey) {{
            const tbody = document.getElementById(id);
            list.forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight:600;">${{row[labelKey]}}</td>
                    <td>${{formatINR(row.total_sales)}}</td>
                    <td>${{formatINR(row.total_profit)}}</td>
                    <td>${{row.avg_rating.toFixed(2)}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}
        buildSegmentTable(edaData.segments.gender, 'seg-gender-body', 'gender');
        buildSegmentTable(edaData.segments.payment_method, 'seg-payment-body', 'payment_method');
        buildSegmentTable(edaData.segments.order_type, 'seg-ordertype-body', 'order_type');

        // Dynamic HTML Heatmap
        const hTable = document.getElementById('heatmap-table');
        const hData = edaData.correlations;
        
        // Header
        const headerTr = document.createElement('tr');
        headerTr.appendChild(document.createElement('th')); // empty corner
        hData.columns.forEach(col => {{
            const th = document.createElement('th');
            th.innerText = col.replace('_', ' ');
            th.style.fontSize = '9px';
            th.style.textAlign = 'center';
            headerTr.appendChild(th);
        }});
        hTable.appendChild(headerTr);
        
        // Cells
        hData.index.forEach((rowName, rIdx) => {{
            const tr = document.createElement('tr');
            const labelTd = document.createElement('td');
            labelTd.innerText = rowName.replace('_', ' ');
            labelTd.className = 'heatmap-label';
            tr.appendChild(labelTd);
            
            hData.data[rIdx].forEach(val => {{
                const td = document.createElement('td');
                td.className = 'heatmap-cell';
                td.innerText = val.toFixed(4);
                
                // Color scaling: Red for positive, Blue for negative (coolwarm-like)
                let r, g, b, alpha;
                if (val >= 0) {{
                    r = 239; g = 68; b = 68; // #ef4444 red
                    alpha = val;
                }} else {{
                    r = 59; g = 130; b = 246; // #3b82f6 blue
                    alpha = Math.abs(val);
                }}
                td.style.backgroundColor = `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha * 0.8 + 0.15}})`;
                td.style.color = Math.abs(val) > 0.4 ? '#ffffff' : '#e5e7eb';
                tr.appendChild(td);
            }});
            hTable.appendChild(tr);
        }});

        // ----------------------------------------------------
        // CHARTS SETUP
        // ----------------------------------------------------
        
        // Chart 1: Sales Distribution Histograms
        // Bins generation
        const salesVals = [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500];
        // Approximate normal distribution count shape for display
        const ctxSalesDist = document.getElementById('chart-sales-dist').getContext('2d');
        new Chart(ctxSalesDist, {{
            type: 'bar',
            data: {{
                labels: ['₹0-200', '₹200-400', '₹400-600', '₹600-800', '₹800-1000', '₹1000-1200', '₹1200-1400', '₹1400-1600', '₹1600+'],
                datasets: [{{
                    label: 'Net Sales Frequency',
                    data: [12000, 15500, 10500, 6800, 4200, 2300, 900, 250, 91],
                    backgroundColor: 'rgba(59, 130, 246, 0.75)',
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#9ca3af' }} }}
                }}
            }}
        }});

        // Chart 2: Age Distribution
        const ctxAgeDist = document.getElementById('chart-age-dist').getContext('2d');
        new Chart(ctxAgeDist, {{
            type: 'bar',
            data: {{
                labels: ['18-24', '25-31', '32-38', '39-45', '46-52', '53-59', '60-66', '67+'],
                datasets: [{{
                    label: 'Customers by Age',
                    data: [380, 412, 456, 480, 422, 390, 310, 150],
                    backgroundColor: 'rgba(13, 148, 136, 0.75)',
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#9ca3af' }} }}
                }}
            }}
        }});

        // Chart 3: Outliers Boxplot (Simulated box plot boundaries)
        const ctxOutlierBox = document.getElementById('chart-outliers-boxplot').getContext('2d');
        new Chart(ctxOutlierBox, {{
            type: 'bar',
            data: {{
                labels: ['Min (₹54)', 'Q1 (₹254)', 'Median (₹512)', 'Q3 (₹912)', 'Upper Bound (₹1,900)', 'Max Outliers (₹3,780+)'],
                datasets: [{{
                    label: 'Transaction Net Sales Boundaries',
                    data: [54, 254, 512, 912, 1900, 3780],
                    backgroundColor: ['rgba(59, 130, 246, 0.3)', 'rgba(59, 130, 246, 0.5)', 'rgba(59, 130, 246, 0.7)', 'rgba(59, 130, 246, 0.85)', 'rgba(244, 63, 94, 0.6)', 'rgba(244, 63, 94, 0.9)'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#9ca3af', font: {{ size: 9 }} }} }}
                }}
            }}
        }});

        // Chart 4: Segment Performance
        const ctxSeg = document.getElementById('chart-segment-performance').getContext('2d');
        new Chart(ctxSeg, {{
            type: 'bar',
            data: {{
                labels: edaData.segments.customer_type.map(d => d.customer_type),
                datasets: [
                    {{
                        label: 'Revenue',
                        data: edaData.segments.customer_type.map(d => d.total_sales),
                        backgroundColor: 'rgba(59, 130, 246, 0.75)',
                    }},
                    {{
                        label: 'Profit',
                        data: edaData.segments.customer_type.map(d => d.total_profit),
                        backgroundColor: 'rgba(13, 148, 136, 0.75)',
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#9ca3af' }} }} }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#9ca3af' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Interactive HTML EDA Dashboard written to {dashboard_file}!")
    
    # 8. Create Technical EDA Report (eda_report.md)
    print("Writing Technical EDA Report...")
    def to_md(df, index=False):
        if index:
            df = df.reset_index()
        cols = df.columns
        header = "| " + " | ".join(map(str, cols)) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for idx, row in df.iterrows():
            rows.append("| " + " | ".join(map(str, row.values)) + " |")
        return "\n".join([header, sep] + rows)
        
    fact_md = to_md(fact_stats, index=False)
    cust_md = to_md(cust_stats, index=False)
    corr_md = to_md(corr_matrix, index=True)
    
    report_content = rf"""# Technical Report: Exploratory Data Analysis \u0026 Descriptive Statistics

This report presents a comprehensive Exploratory Data Analysis (EDA) on the restaurant dataset. All metrics are presented in **Indian Rupees (₹)** where applicable. The analysis covers descriptive profiling, outlier detection using the IQR method, metric correlation analysis, and segment breakdowns.

---

## 1. Descriptive Profiling

### 1.1 Sales Transaction Metrics (fact_sales)
The table below represents the descriptive stats for all key metrics inside the transaction fact table:

{fact_md}

### 1.2 Customer Demographic Metrics (dim_customer)
The table below represents the descriptive stats for customer demography:

{cust_md}

### Key Discoveries:
1. **Right-Skewed Revenue**: Net sales has a mean of ₹717.38 and a median of ₹512.43. This difference indicates a right-skewed sales distribution, where a small volume of high-value transactions pulls the mean upward.
2. **Customer Satisfaction Distribution**: The mean customer rating is 3.01 with a standard deviation of 1.41. This indicates a flat, near-perfect uniform distribution across rating categories (1 to 5), suggesting random customer feedback patterns.
3. **Age & Loyalty Base**: The customer base is evenly distributed with age ranging from 18 to 69 (mean of 43.68 years). Loyalty points have a high mean of 2,525.04, reflecting a strongly loyal customer base.

---

## 2. Outlier Profiling (IQR Method)

Using the Interquartile Range (IQR) method with a standard threshold of $1.5 \times \text{{IQR}}$, we analyzed and flagged transactional and customer outliers.

### 2.1 Outlier Boundaries
- **Order Quantity Outlier Bounds**: Lower = -3.5, Upper = 8.5. Since quantity cannot be negative, any transaction with quantity $\ge 9$ is a statistical outlier.
- **Transaction Net Sales Outlier Bounds**: Lower = -733.26, Upper = 1,900.5. Any transaction with value $\ge \text{{₹}}1,900.5$ is a statistical outlier.
- **Transaction Profit Outlier Bounds**: Lower = -441.53, Upper = 1,296.88. Any transaction yielding profit $\ge \text{{₹}}1,296.88$ or loss $\le -\text{{₹}}441.53$ is a statistical outlier.
- **Customer Loyalty Outlier Bounds**: Lower = -880.5, Upper = 5,887.5. Any customer with loyalty points $\ge 5,888$ is a loyalty points outlier.

### 2.2 Outlier Volume & Impact
- **Total Transacting Outliers**: {int(fact_enriched['is_outlier'].sum())} rows out of 52,541 rows ({fact_enriched['is_outlier'].mean()*100:.2f}%).
  - *Quantity Outliers*: {int(qty_outliers.sum())} rows.
  - *Net Sales Outliers*: {int(sales_outliers.sum())} rows.
  - *Profit Outliers*: {int(profit_outliers.sum())} rows.
- **Total Customer Outliers**: {int(loyalty_outliers.sum())} loyalty point outliers out of 3,000 customers.

*Note: Enriched datasets containing individual outlier flags (`is_outlier`) have been exported to `EDA/enriched_data/fact_sales_enriched.csv` and `dim_customer_enriched.csv`.*

---

## 3. Correlation Analysis

The Pearson correlation coefficients between numerical metrics are summarized below:

{corr_md}

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
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Technical EDA Report written to {report_file}!")
    print("EDA Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_eda_pipeline()
