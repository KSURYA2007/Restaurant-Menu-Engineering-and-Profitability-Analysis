import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations():
    print("Starting visualization generation...")
    
    # 1. Paths
    cleaned_dir = "d:/cat/DAL/project/ETL/cleaned_data"
    viz_dir = "d:/cat/DAL/project/ETL/visualizations"
    dashboard_file = "d:/cat/DAL/project/ETL/dashboard.html"
    
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load cleaned data
    fact_sales = pd.read_csv(os.path.join(cleaned_dir, 'fact_sales.csv'))
    dim_menu = pd.read_csv(os.path.join(cleaned_dir, 'dim_menu.csv'))
    dim_location = pd.read_csv(os.path.join(cleaned_dir, 'dim_location.csv'))
    dim_date = pd.read_csv(os.path.join(cleaned_dir, 'dim_date.csv'))
    dim_customer = pd.read_csv(os.path.join(cleaned_dir, 'dim_customer.csv'))
    
    # Match datetime
    fact_sales['order_date'] = pd.to_datetime(fact_sales['order_date'])
    dim_date['date'] = pd.to_datetime(dim_date['date'])
    
    # Set seaborn style
    sns.set_theme(style="darkgrid")
    
    # ----------------------------------------------------
    # STATIC CHART 1: BCG Menu Engineering Scatter Plot
    # ----------------------------------------------------
    print("Generating Static Chart 1: BCG Scatter Plot...")
    plt.figure(figsize=(12, 8))
    
    # Benchmarks
    avg_popularity = dim_menu['total_quantity_sold'].mean()
    avg_profitability = dim_menu['unit_contribution_margin'].mean()
    
    colors = {'Star': '#10b981', 'Plowhorse': '#3b82f6', 'Puzzle': '#f59e0b', 'Dog': '#f43f5e'}
    
    sns.scatterplot(
        data=dim_menu,
        x='total_quantity_sold',
        y='unit_contribution_margin',
        hue='menu_category',
        palette=colors,
        s=150,
        alpha=0.8,
        edgecolor='w',
        linewidth=1.5
    )
    
    # Add quad lines
    plt.axvline(x=avg_popularity, color='#64748b', linestyle='--', linewidth=1.5, label=f'Avg Vol ({avg_popularity:.1f})')
    plt.axhline(y=avg_profitability, color='#64748b', linestyle='--', linewidth=1.5, label=f'Avg Margin (₹{avg_profitability:.1f})')
    
    # Annotate items
    for idx, row in dim_menu.iterrows():
        plt.text(
            row['total_quantity_sold'] + 30,
            row['unit_contribution_margin'],
            row['item_name'],
            fontsize=9,
            alpha=0.8,
            verticalalignment='center'
        )
        
    plt.title('Menu Engineering Matrix (BCG Product Portfolio)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Popularity (Total Quantity Sold)', fontsize=12, labelpad=10)
    plt.ylabel('Profitability (Unit Contribution Margin - ₹)', fontsize=12, labelpad=10)
    plt.legend(title='BCG Category', loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'menu_engineering_bcg_matrix.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # STATIC CHART 2: Profit Contribution by BCG Category
    # ----------------------------------------------------
    print("Generating Static Chart 2: Profit by BCG Category...")
    plt.figure(figsize=(10, 6))
    
    # Aggregate profit by category
    sales_merged = fact_sales.merge(dim_menu[['item_id', 'menu_category']], on='item_id')
    profit_by_cat = sales_merged.groupby('menu_category')['profit'].sum().reset_index()
    
    # Sort for aesthetics
    profit_by_cat = profit_by_cat.sort_values(by='profit', ascending=False)
    
    sns.barplot(
        data=profit_by_cat,
        x='menu_category',
        y='profit',
        palette=colors,
        hue='menu_category',
        legend=False
    )
    
    plt.title('Total Profit Contribution by Menu Category', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Menu Category (BCG Classification)', fontsize=12, labelpad=10)
    plt.ylabel('Total Profit (₹)', fontsize=12, labelpad=10)
    
    # Add values on top of bars
    for index, row in profit_by_cat.iterrows():
        plt.text(
            index, 
            row['profit'] + (profit_by_cat['profit'].max() * 0.01), 
            f"₹{row['profit']:,.2f}", 
            color='black', 
            ha="center", 
            fontsize=10, 
            fontweight='bold'
        )
        
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'profit_by_bcg_category.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # STATIC CHART 3: Monthly Net Sales and Profit Trend
    # ----------------------------------------------------
    print("Generating Static Chart 3: Monthly Sales Trend...")
    plt.figure(figsize=(12, 6))
    
    # Group sales by year and month
    fact_sales['year_month'] = fact_sales['order_date'].dt.to_period('M')
    monthly_stats = fact_sales.groupby('year_month').agg(
        net_sales=('net_sales', 'sum'),
        profit=('profit', 'sum')
    ).reset_index()
    monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)
    
    plt.plot(monthly_stats['year_month'], monthly_stats['net_sales'], marker='o', color='#3b82f6', linewidth=2.5, label='Net Sales')
    plt.plot(monthly_stats['year_month'], monthly_stats['profit'], marker='s', color='#10b981', linewidth=2.5, label='Profit')
    
    plt.title('Monthly Net Sales & Profit Trend (2022 - 2024)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Month', fontsize=12, labelpad=10)
    plt.ylabel('Amount (₹)', fontsize=12, labelpad=10)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'monthly_sales_trend.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # STATIC CHART 4: Net Sales and Profit by Location
    # ----------------------------------------------------
    print("Generating Static Chart 4: Sales by Location...")
    plt.figure(figsize=(10, 6))
    
    loc_merged = fact_sales.merge(dim_location, on='location_id')
    loc_stats = loc_merged.groupby('branch_name').agg(
        net_sales=('net_sales', 'sum'),
        profit=('profit', 'sum')
    ).reset_index().melt(id_vars='branch_name', value_vars=['net_sales', 'profit'], var_name='Metric', value_name='Value')
    
    sns.barplot(
        data=loc_stats,
        x='branch_name',
        y='Value',
        hue='Metric',
        palette={'net_sales': '#3b82f6', 'profit': '#10b981'}
    )
    
    plt.title('Performance Breakdown by Branch/Location', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Branch', fontsize=12, labelpad=10)
    plt.ylabel('Amount (₹)', fontsize=12, labelpad=10)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'branch_performance.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # GENERATE INTERACTIVE HTML DASHBOARD
    # ----------------------------------------------------
    print("Generating Interactive HTML Dashboard...")
    
    # Compute aggregates for dashboard JSON embedding
    kpis = {
        'total_revenue': float(fact_sales['net_sales'].sum()),
        'total_profit': float(fact_sales['profit'].sum()),
        'total_orders': int(fact_sales['order_id'].nunique()),
        'total_items_sold': int(fact_sales['quantity'].sum()),
        'average_rating': float(fact_sales['customer_rating'].mean()),
        'profit_margin': float(fact_sales['profit'].sum() / fact_sales['net_sales'].sum() * 100),
    }
    
    # Monthly trend JSON
    monthly_trend_json = monthly_stats.to_dict(orient='records')
    
    # Branch performance JSON
    branch_merged = fact_sales.merge(dim_location, on='location_id')
    branch_perf = branch_merged.groupby(['branch_name', 'city', 'region']).agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        orders=('order_id', 'nunique'),
        qty_sold=('quantity', 'sum')
    ).reset_index().to_dict(orient='records')
    
    # Menu classification summary JSON
    menu_cat_perf = sales_merged.groupby('menu_category').agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        qty_sold=('quantity', 'sum')
    ).reset_index().to_dict(orient='records')
    
    # Menu Items detail list JSON
    menu_items_detail = dim_menu.copy()
    menu_items_detail['total_revenue'] = menu_items_detail['item_id'].map(fact_sales.groupby('item_id')['net_sales'].sum())
    menu_items_detail['total_profit'] = menu_items_detail['item_id'].map(fact_sales.groupby('item_id')['profit'].sum())
    menu_items_detail['avg_rating'] = menu_items_detail['item_id'].map(fact_sales.groupby('item_id')['customer_rating'].mean())
    menu_items_detail = menu_items_detail.round(2).fillna(0)
    menu_items_json = menu_items_detail.to_dict(orient='records')
    
    # Payment and Order type JSON
    payment_stats = fact_sales.groupby('payment_method').agg(
        revenue=('net_sales', 'sum'),
        orders=('order_id', 'nunique')
    ).reset_index().to_dict(orient='records')
    
    order_type_stats = fact_sales.groupby('order_type').agg(
        revenue=('net_sales', 'sum'),
        orders=('order_id', 'nunique')
    ).reset_index().to_dict(orient='records')
    
    # Customer type distribution
    customer_stats = dim_customer.groupby('customer_type').agg(
        count=('customer_id', 'count'),
        avg_loyalty=('loyalty_points', 'mean')
    ).reset_index().to_dict(orient='records')

    # Bundle all data
    dashboard_data = {
        'kpis': kpis,
        'monthly_trend': monthly_trend_json,
        'branch_performance': branch_perf,
        'menu_category_performance': menu_cat_perf,
        'menu_items': menu_items_json,
        'payment_stats': payment_stats,
        'order_type_stats': order_type_stats,
        'customer_stats': customer_stats,
        'benchmarks': {
            'popularity': avg_popularity,
            'profitability': avg_profitability
        }
    }
    
    # Read/write HTML
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Menu Engineering & Profitability Analysis Dashboard</title>
    
    <!-- Outfit & Inter Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {{
            --bg-base: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border-card: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            
            /* BCG Colors */
            --color-star: #10b981;
            --color-plowhorse: #3b82f6;
            --color-puzzle: #f59e0b;
            --color-dog: #f43f5e;
            
            /* Theme Accents */
            --accent-purple: #8b5cf6;
            --accent-teal: #14b8a6;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
        }}
        
        body {{
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 10% 20%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(20, 184, 166, 0.12) 0px, transparent 50%);
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
        
        /* Glassmorphic Styles */
        .glass-panel {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 24px;
            transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }}
        
        .glass-panel:hover {{
            background: var(--bg-card-hover);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }}
        
        /* Layout Structure */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
            background: rgba(15, 23, 42, 0.6);
            border-bottom: 1px solid var(--border-card);
            margin: -24px -24px 0 -24px;
            backdrop-filter: blur(8px);
        }}
        
        header h1 {{
            font-size: 28px;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header-meta {{
            font-size: 14px;
            color: var(--text-secondary);
            text-align: right;
        }}
        
        .tab-navigation {{
            display: flex;
            gap: 12px;
            margin-bottom: 8px;
        }}
        
        .tab-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            padding: 10px 20px;
            border-radius: 9999px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        
        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .tab-btn.active {{
            background: var(--accent-purple);
            color: white;
            border-color: var(--accent-purple);
            box-shadow: 0 0 16px rgba(139, 92, 246, 0.4);
        }}
        
        /* Dashboard Views */
        .dashboard-view {{
            display: none;
            flex-direction: column;
            gap: 24px;
        }}
        
        .dashboard-view.active {{
            display: flex;
        }}
        
        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
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
            background: var(--accent-teal);
            border-radius: 4px 0 0 4px;
        }}
        
        .kpi-card.purple::before {{
            background: var(--accent-purple);
        }}
        
        .kpi-card.star::before {{
            background: var(--color-star);
        }}
        
        .kpi-card.puzzle::before {{
            background: var(--color-puzzle);
        }}
        
        .kpi-title {{
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .kpi-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 32px;
            font-weight: 800;
            color: var(--text-primary);
        }}
        
        /* Grid Layouts */
        .dashboard-row-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 24px;
        }}
        
        .dashboard-row-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
        }}
        
        .chart-container {{
            position: relative;
            height: 320px;
            width: 100%;
        }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge.star {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--color-star);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        
        .badge.plowhorse {{
            background: rgba(59, 130, 246, 0.15);
            color: var(--color-plowhorse);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        
        .badge.puzzle {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--color-puzzle);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        
        .badge.dog {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--color-dog);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }}
        
        .badge.veg {{
            background: rgba(20, 184, 166, 0.15);
            color: var(--accent-teal);
            border: 1px solid rgba(20, 184, 166, 0.3);
        }}
        
        .badge.nonveg {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--color-dog);
            border: 1px solid rgba(244, 63, 94, 0.3);
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
            font-size: 14px;
        }}
        
        th, td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        th {{
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}
        
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        
        /* Menu filter controls */
        .filter-controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 15px;
        }}
        
        .search-input {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            color: white;
            padding: 8px 16px;
            font-size: 14px;
            outline: none;
            width: 250px;
            transition: all 0.2s ease;
        }}
        
        .search-input:focus {{
            border-color: var(--accent-purple);
            box-shadow: 0 0 8px rgba(139, 92, 246, 0.2);
        }}
        
        .filter-select {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            color: white;
            padding: 8px 12px;
            font-size: 14px;
            outline: none;
            cursor: pointer;
        }}
        
        .legend-container {{
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid var(--border-card);
            font-size: 13px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            flex: 1;
            padding: 0 10px;
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-top: 3px;
            flex-shrink: 0;
        }}
        
        .legend-text {{
            color: var(--text-secondary);
            line-height: 1.4;
        }}
        
        .legend-name {{
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 2px;
        }}
    </style>
</head>
<body>

    <!-- Header Section -->
    <header>
        <div>
            <h1>Restaurant Menu Analysis</h1>
            <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">Premium Analytics Dashboard & Menu Engineering (Star Schema Model)</p>
        </div>
        <div class="header-meta">
            <div style="font-weight: 600;">Dataset Status: <span style="color: var(--accent-teal)">Cleaned & Verified</span></div>
            <div style="font-size: 12px; margin-top: 4px;">Pipeline Engine: Python Pandas/Star Schema</div>
        </div>
    </header>

    <!-- Tab navigation -->
    <div class="tab-navigation">
        <button class="tab-btn active" onclick="switchTab('overview')">Overview Dashboard</button>
        <button class="tab-btn" onclick="switchTab('menu-engineering')">Menu Engineering (BCG)</button>
        <button class="tab-btn" onclick="switchTab('branch-demographics')">Branch & Customer Insights</button>
    </div>

    <!-- VIEW 1: OVERVIEW -->
    <div id="view-overview" class="dashboard-view active">
        <!-- KPI Row -->
        <div class="kpi-grid">
            <div class="glass-panel kpi-card">
                <div class="kpi-title">Net Sales</div>
                <div class="kpi-value" id="kpi-sales">₹0.00</div>
            </div>
            <div class="glass-panel kpi-card purple">
                <div class="kpi-title">Total Profit</div>
                <div class="kpi-value" id="kpi-profit">₹0.00</div>
            </div>
            <div class="glass-panel kpi-card star">
                <div class="kpi-title">Profit Margin</div>
                <div class="kpi-value" id="kpi-margin">0.0%</div>
            </div>
            <div class="glass-panel kpi-card puzzle">
                <div class="kpi-title">Total Orders</div>
                <div class="kpi-value" id="kpi-orders">0</div>
            </div>
        </div>

        <!-- Row 2: Monthly Trends & Location -->
        <div class="dashboard-row-2">
            <div class="glass-panel">
                <h3 style="margin-bottom: 15px;">Monthly Sales & Profit Trend</h3>
                <div class="chart-container">
                    <canvas id="chart-monthly-trend"></canvas>
                </div>
            </div>
            
            <div class="glass-panel">
                <h3 style="margin-bottom: 15px;">Branch Contribution Breakdown</h3>
                <div class="chart-container">
                    <canvas id="chart-branch-performance"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Row 3: Payment & Order Type Distribution -->
        <div class="dashboard-row-3">
            <div class="glass-panel">
                <h3 style="margin-bottom: 15px;">Order Type Performance</h3>
                <div class="chart-container" style="height: 220px;">
                    <canvas id="chart-order-types"></canvas>
                </div>
            </div>
            <div class="glass-panel">
                <h3 style="margin-bottom: 15px;">Payment Method Distribution</h3>
                <div class="chart-container" style="height: 220px;">
                    <canvas id="chart-payments"></canvas>
                </div>
            </div>
            <div class="glass-panel">
                <h3 style="margin-bottom: 15px;">Menu Category Performance</h3>
                <div class="chart-container" style="height: 220px;">
                    <canvas id="chart-menu-categories"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- VIEW 2: MENU ENGINEERING (BCG MATRIX) -->
    <div id="view-menu-engineering" class="dashboard-view">
        <div class="dashboard-row-2">
            <!-- BCG Scatter Plot Chart -->
            <div class="glass-panel" style="flex: 1.3;">
                <h3 style="margin-bottom: 5px;">Interactive BCG Menu Engineering Matrix</h3>
                <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 15px;">
                    Hover over dots to see item names, volume sold, and unit margins. Quad lines denote category averages.
                </p>
                <div class="chart-container" style="height: 400px;">
                    <canvas id="chart-bcg-scatter"></canvas>
                </div>
                
                <!-- BCG Quadrant Legend / Explanation -->
                <div class="legend-container">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: var(--color-star);"></div>
                        <div class="legend-text">
                            <div class="legend-name" style="color: var(--color-star);">Stars</div>
                            <div>High popularity, High profitability. Keep these prominent and maintain quality.</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: var(--color-plowhorse);"></div>
                        <div class="legend-text">
                            <div class="legend-name" style="color: var(--color-plowhorse);">Plowhorses</div>
                            <div>High popularity, Low profitability. Try slightly raising prices or reducing cost sizes.</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: var(--color-puzzle);"></div>
                        <div class="legend-text">
                            <div class="legend-name" style="color: var(--color-puzzle);">Puzzles</div>
                            <div>Low popularity, High profitability. Improve marketing or menu placement.</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: var(--color-dog);"></div>
                        <div class="legend-text">
                            <div class="legend-name" style="color: var(--color-dog);">Dogs</div>
                            <div>Low popularity, Low profitability. Consider replacing or redesigning these items.</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- BCG Category Profit distribution -->
            <div class="glass-panel" style="flex: 0.7; display: flex; flex-direction: column;">
                <h3 style="margin-bottom: 15px;">Profit Distribution by BCG Class</h3>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="chart-bcg-profit"></canvas>
                </div>
                <div style="margin-top: 20px; font-size: 13px; color: var(--text-secondary);">
                    <h4 style="color: white; margin-bottom: 8px; font-size: 14px;">Strategic Takeaway</h4>
                    <p style="line-height: 1.4; margin-bottom: 8px;">
                        <strong>Stars</strong> and <strong>Puzzles</strong> provide premium margins per unit, while <strong>Plowhorses</strong> drive volume and cash flow.
                    </p>
                    <p style="line-height: 1.4;">
                        Target <strong>Dogs</strong> and <strong>Plowhorses</strong> for recipe costing optimization to improve the overall blend margin.
                    </p>
                </div>
            </div>
        </div>

        <!-- Menu Item Detail Table -->
        <div class="glass-panel">
            <div class="filter-controls">
                <h3>Menu Items Engineering Performance List</h3>
                <div style="display: flex; gap: 12px;">
                    <input type="text" id="table-search" class="search-input" placeholder="Search item name..." oninput="filterTable()">
                    <select id="table-filter-cat" class="filter-select" onchange="filterTable()">
                        <option value="All">All BCG Categories</option>
                        <option value="Star">Stars</option>
                        <option value="Plowhorse">Plowhorses</option>
                        <option value="Puzzle">Puzzles</option>
                        <option value="Dog">Dogs</option>
                    </select>
                </div>
            </div>
            
            <div class="table-wrapper">
                <table id="menu-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Item Name</th>
                            <th>Category</th>
                            <th>Veg?</th>
                            <th>Base Price</th>
                            <th>Base Cost</th>
                            <th>Units Sold</th>
                            <th>Unit Margin</th>
                            <th>Total Profit</th>
                            <th>BCG Status</th>
                        </tr>
                    </thead>
                    <tbody id="menu-table-body">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- VIEW 3: BRANCH & CUSTOMER DEMOGRAPHICS -->
    <div id="view-branch-demographics" class="dashboard-view">
        <div class="dashboard-row-2">
            <div class="glass-panel">
                <h3 style="margin-bottom: 15px;">Branch Sales Breakdown</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Branch</th>
                                <th>City</th>
                                <th>Region</th>
                                <th>Revenue</th>
                                <th>Profit</th>
                                <th>Orders</th>
                                <th>Qty Sold</th>
                            </tr>
                        </thead>
                        <tbody id="branch-table-body">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="glass-panel">
                <h3 style="margin-bottom: 15px;">Customer Type Breakdown</h3>
                <div class="chart-container" style="height: 300px;">
                    <canvas id="chart-customers"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embed compiled data from python script
        const data = {json.dumps(dashboard_data, indent=4)};
        
        // Tab switching
        function switchTab(tabId) {{
            // Deactivate all
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.dashboard-view').forEach(view => view.classList.remove('active'));
            
            // Activate target
            let activeBtn = document.querySelector(`.tab-btn[onclick="switchTab('${{tabId}}')"]`);
            if (activeBtn) activeBtn.classList.add('active');
            
            let targetView = document.getElementById(`view-${{tabId}}`);
            if (targetView) targetView.classList.add('active');
        }}

        // Format currency helper
        const formatCurrency = (val) => {{
            return new Intl.NumberFormat('en-IN', {{ style: 'currency', currency: 'INR' }}).format(val);
        }};
        
        // Format integer helper
        const formatNumber = (val) => {{
            return new Intl.NumberFormat('en-US').format(val);
        }};

        // Initialize KPIs
        document.getElementById('kpi-sales').innerText = formatCurrency(data.kpis.total_revenue);
        document.getElementById('kpi-profit').innerText = formatCurrency(data.kpis.total_profit);
        document.getElementById('kpi-margin').innerText = data.kpis.profit_margin.toFixed(2) + "%";
        document.getElementById('kpi-orders').innerText = formatNumber(data.kpis.total_orders);

        // Build Table rows
        const tableBody = document.getElementById('menu-table-body');
        data.menu_items.forEach(item => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${{item.item_id}}</td>
                <td style="font-weight: 600;">${{item.item_name}}</td>
                <td>${{item.category_name}}</td>
                <td><span class="badge ${{item.is_vegetarian === 'Yes' ? 'veg' : 'nonveg'}}">${{item.is_vegetarian}}</span></td>
                <td>${{formatCurrency(item.base_price)}}</td>
                <td>${{formatCurrency(item.base_cost)}}</td>
                <td>${{formatNumber(item.total_quantity_sold)}}</td>
                <td>${{formatCurrency(item.unit_contribution_margin)}}</td>
                <td style="font-weight: 600; color: ${{item.total_profit >= 0 ? '#10b981' : '#f43f5e'}};">${{formatCurrency(item.total_profit)}}</td>
                <td><span class="badge ${{item.menu_category.toLowerCase()}}">${{item.menu_category}}</span></td>
            `;
            tableBody.appendChild(tr);
        }});
        
        // Branch Table
        const branchBody = document.getElementById('branch-table-body');
        data.branch_performance.forEach(b => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: 600;">${{b.branch_name}}</td>
                <td>${{b.city}}</td>
                <td>${{b.region}}</td>
                <td>${{formatCurrency(b.revenue)}}</td>
                <td style="font-weight: 600; color: #10b981;">${{formatCurrency(b.profit)}}</td>
                <td>${{formatNumber(b.orders)}}</td>
                <td>${{formatNumber(b.qty_sold)}}</td>
            `;
            branchBody.appendChild(tr);
        }});

        // Filter Table Function
        function filterTable() {{
            const query = document.getElementById('table-search').value.toLowerCase();
            const categoryFilter = document.getElementById('table-filter-cat').value;
            const rows = document.querySelectorAll('#menu-table-body tr');
            
            rows.forEach(row => {{
                const itemName = row.children[1].innerText.toLowerCase();
                const bcgStatus = row.children[9].innerText.trim();
                
                const matchesSearch = itemName.includes(query);
                const matchesCategory = categoryFilter === 'All' || bcgStatus === categoryFilter;
                
                if (matchesSearch && matchesCategory) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}

        // ----------------------------------------------------
        // CHARTS SETUP
        // ----------------------------------------------------
        
        // 1. Monthly Trend Chart
        const ctxMonthly = document.getElementById('chart-monthly-trend').getContext('2d');
        new Chart(ctxMonthly, {{
            type: 'line',
            data: {{
                labels: data.monthly_trend.map(d => d.year_month),
                datasets: [
                    {{
                        label: 'Net Sales',
                        data: data.monthly_trend.map(d => d.net_sales),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.3
                    }},
                    {{
                        label: 'Profit',
                        data: data.monthly_trend.map(d => d.profit),
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#94a3b8' }} }}
                }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // 2. Branch Performance Chart
        const ctxBranch = document.getElementById('chart-branch-performance').getContext('2d');
        new Chart(ctxBranch, {{
            type: 'bar',
            data: {{
                labels: data.branch_performance.map(b => b.branch_name),
                datasets: [
                    {{
                        label: 'Revenue',
                        data: data.branch_performance.map(b => b.revenue),
                        backgroundColor: 'rgba(59, 130, 246, 0.75)',
                        borderColor: '#3b82f6',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Profit',
                        data: data.branch_performance.map(b => b.profit),
                        backgroundColor: 'rgba(16, 185, 129, 0.75)',
                        borderColor: '#10b981',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#94a3b8' }} }}
                }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // 3. Order Types Doughnut
        const ctxOrderTypes = document.getElementById('chart-order-types').getContext('2d');
        new Chart(ctxOrderTypes, {{
            type: 'doughnut',
            data: {{
                labels: data.order_type_stats.map(d => d.order_type),
                datasets: [{{
                    data: data.order_type_stats.map(d => d.revenue),
                    backgroundColor: ['#8b5cf6', '#14b8a6', '#f59e0b'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}
                }}
            }}
        }});

        // 4. Payments Doughnut
        const ctxPayments = document.getElementById('chart-payments').getContext('2d');
        new Chart(ctxPayments, {{
            type: 'doughnut',
            data: {{
                labels: data.payment_stats.map(d => d.payment_method),
                datasets: [{{
                    data: data.payment_stats.map(d => d.revenue),
                    backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ size: 10 }} }} }}
                }}
            }}
        }});

        // 5. Menu Categories Bar
        const ctxMenuCat = document.getElementById('chart-menu-categories').getContext('2d');
        new Chart(ctxMenuCat, {{
            type: 'bar',
            data: {{
                labels: data.menu_category_performance.map(d => d.menu_category),
                datasets: [{{
                    label: 'Net Sales',
                    data: data.menu_category_performance.map(d => d.revenue),
                    backgroundColor: '#8b5cf6',
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#94a3b8', font: {{ size: 9 }} }} }},
                    x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#94a3b8', font: {{ size: 9 }} }} }}
                }}
            }}
        }});

        // 6. BCG Scatter Plot Chart (Interactive)
        const ctxBcg = document.getElementById('chart-bcg-scatter').getContext('2d');
        const bcgPoints = data.menu_items.map(item => ({{
            x: item.total_quantity_sold,
            y: item.unit_contribution_margin,
            label: item.item_name,
            category: item.menu_category
        }}));

        // Group points by category for coloring
        const scatterDatasets = ['Star', 'Plowhorse', 'Puzzle', 'Dog'].map(cat => {{
            const catColors = {{ 'Star': '#10b981', 'Plowhorse': '#3b82f6', 'Puzzle': '#f59e0b', 'Dog': '#f43f5e' }};
            return {{
                label: cat + 's',
                data: bcgPoints.filter(p => p.category === cat),
                backgroundColor: catColors[cat],
                pointRadius: 8,
                pointHoverRadius: 10,
            }};
        }});

        new Chart(ctxBcg, {{
            type: 'scatter',
            data: {{
                datasets: [
                    ...scatterDatasets,
                    // Horizontal benchmark line
                    {{
                        label: 'Avg Margin',
                        data: [
                            {{ x: 0, y: data.benchmarks.profitability }},
                            {{ x: Math.max(...bcgPoints.map(p => p.x)) * 1.1, y: data.benchmarks.profitability }}
                        ],
                        type: 'line',
                        borderColor: 'rgba(255, 255, 255, 0.25)',
                        borderWidth: 1.5,
                        borderDash: [6, 6],
                        pointRadius: 0,
                        fill: false,
                        showLine: true
                    }},
                    // Vertical benchmark line
                    {{
                        label: 'Avg Quantity',
                        data: [
                            {{ x: data.benchmarks.popularity, y: 0 }},
                            {{ x: data.benchmarks.popularity, y: Math.max(...bcgPoints.map(p => p.y)) * 1.1 }}
                        ],
                        type: 'line',
                        borderColor: 'rgba(255, 255, 255, 0.25)',
                        borderWidth: 1.5,
                        borderDash: [6, 6],
                        pointRadius: 0,
                        fill: false,
                        showLine: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ 
                            color: '#94a3b8',
                            filter: function(item, chartData) {{
                                return item.text !== 'Avg Margin' && item.text !== 'Avg Quantity';
                            }}
                        }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const raw = context.raw;
                                return raw.label ? `${{raw.label}}: Vol=${{raw.x}}, Margin=${{formatCurrency(raw.y)}}` : '';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Popularity (Total Quantity Sold)', color: '#94a3b8' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Unit Contribution Margin (₹)', color: '#94a3b8' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});

        // 7. BCG Profit Bar Chart
        const ctxBcgProfit = document.getElementById('chart-bcg-profit').getContext('2d');
        const sortedBcgStats = [...data.menu_category_performance].sort((a,b) => b.profit - a.profit);
        new Chart(ctxBcgProfit, {{
            type: 'bar',
            data: {{
                labels: sortedBcgStats.map(d => d.menu_category + 's'),
                datasets: [{{
                    data: sortedBcgStats.map(d => d.profit),
                    backgroundColor: sortedBcgStats.map(d => {{
                        const catColors = {{ 'Star': '#10b981', 'Plowhorse': '#3b82f6', 'Puzzle': '#f59e0b', 'Dog': '#f43f5e' }};
                        return catColors[d.menu_category];
                    }}),
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // 8. Customer Demographic Loyalty Polar Area Chart
        const ctxCust = document.getElementById('chart-customers').getContext('2d');
        new Chart(ctxCust, {{
            type: 'polarArea',
            data: {{
                labels: data.customer_stats.map(d => d.customer_type),
                datasets: [{{
                    label: 'Avg Loyalty Points',
                    data: data.customer_stats.map(d => d.avg_loyalty),
                    backgroundColor: [
                        'rgba(139, 92, 246, 0.65)',
                        'rgba(20, 184, 166, 0.65)',
                        'rgba(59, 130, 246, 0.65)',
                        'rgba(245, 158, 11, 0.65)'
                    ],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#94a3b8' }} }}
                }},
                scales: {{
                    r: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        angleLines: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ display: false }}
                    }}
                }}
            }}
        }});

    </script>
</body>
</html>
"""
    
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    print(f"Interactive HTML dashboard written to {dashboard_file}!")
    print("Visualization Generation Completed Successfully!")

if __name__ == "__main__":
    generate_visualizations()
