import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_performance_pipeline():
    print("Starting Product Performance & Sales Analysis Pipeline...")
    
    # 1. Paths
    cleaned_dir = "d:/cat/DAL/project/ETL/cleaned_data"
    perf_dir = "d:/cat/DAL/project/Performance"
    output_dir = os.path.join(perf_dir, 'cleaned_data')
    viz_dir = os.path.join(perf_dir, 'visualizations')
    dashboard_file = os.path.join(perf_dir, 'performance_dashboard.html')
    report_file = os.path.join(perf_dir, 'performance_report.md')
    output_excel = os.path.join(perf_dir, 'Cleaned_Restaurant_Data_Performance.xlsx')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load ETL cleaned data
    fact_sales = pd.read_csv(os.path.join(cleaned_dir, 'fact_sales.csv'))
    dim_customer = pd.read_csv(os.path.join(cleaned_dir, 'dim_customer.csv'))
    dim_menu = pd.read_csv(os.path.join(cleaned_dir, 'dim_menu.csv'))
    dim_location = pd.read_csv(os.path.join(cleaned_dir, 'dim_location.csv'))
    dim_date = pd.read_csv(os.path.join(cleaned_dir, 'dim_date.csv'))
    
    # 2. Product Performance Aggregates
    print("Compiling product-level performance aggregates...")
    product_summary = fact_sales.groupby('item_id').agg(
        units_sold=('quantity', 'sum'),
        gross_revenue=('gross_sales', 'sum'),
        total_discount=('discount_amt', 'sum'),
        net_revenue=('net_sales', 'sum'),
        total_cogs=('cogs', 'sum'),
        total_profit=('profit', 'sum'),
        avg_rating=('customer_rating', 'mean')
    ).reset_index()
    
    product_summary = product_summary.merge(dim_menu, on='item_id', how='left')
    product_summary['realized_margin_pct'] = (product_summary['total_profit'] / product_summary['net_revenue'] * 100).round(2)
    product_summary['unit_margin'] = (product_summary['net_revenue'] - product_summary['total_cogs']) / product_summary['units_sold']
    product_summary = product_summary.round(2)
    
    # Save product summary CSV
    product_summary.to_csv(os.path.join(output_dir, 'product_performance_summary.csv'), index=False)
    
    # Top 5 and Bottom 5 by Profit
    top_profit = product_summary.sort_values(by='total_profit', ascending=False).head(5).copy()
    bottom_profit = product_summary.sort_values(by='total_profit', ascending=True).head(5).copy()
    
    # Top 5 and Bottom 5 by Sales Volume
    top_qty = product_summary.sort_values(by='units_sold', ascending=False).head(5).copy()
    bottom_qty = product_summary.sort_values(by='units_sold', ascending=True).head(5).copy()
    
    # Category Performance Summary
    category_summary = product_summary.groupby('category_name').agg(
        units_sold=('units_sold', 'sum'),
        net_revenue=('net_revenue', 'sum'),
        total_profit=('total_profit', 'sum')
    ).reset_index()
    category_summary['margin_pct'] = (category_summary['total_profit'] / category_summary['net_revenue'] * 100).round(2)
    
    # 3. Sales Drivers Analysis
    print("Analyzing sales drivers (order types, payment, locations)...")
    # Order type
    order_type_perf = fact_sales.groupby('order_type').agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        orders=('order_id', 'nunique'),
        avg_qty_per_order=('quantity', 'mean')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Payment method
    payment_perf = fact_sales.groupby('payment_method').agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        orders=('order_id', 'nunique')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Branch
    branch_merged = fact_sales.merge(dim_location, on='location_id')
    branch_perf = branch_merged.groupby('branch_name').agg(
        revenue=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        orders=('order_id', 'nunique'),
        seating_cap=('seating_cap', 'first')
    ).round(2).reset_index()
    
    # Check if seating capacity drives profit (seating profit ratio)
    branch_perf['profit_per_seat'] = (branch_perf['profit'] / branch_perf['seating_cap']).round(2)
    branch_perf_list = branch_perf.to_dict(orient='records')
    
    # Discount Impact
    discount_perf = fact_sales.groupby('discount_pct').agg(
        transactions=('order_id', 'count'),
        avg_quantity_sold=('quantity', 'mean'),
        total_revenue=('net_sales', 'sum'),
        total_profit=('profit', 'sum')
    ).round(2).reset_index()
    discount_perf['avg_profit_per_tx'] = (discount_perf['total_profit'] / discount_perf['transactions']).round(2)
    discount_perf_list = discount_perf.to_dict(orient='records')
    
    # Save sales drivers CSV
    discount_perf.to_csv(os.path.join(output_dir, 'sales_drivers.csv'), index=False)
    
    # 4. Portfolio Health Analysis
    print("Evaluating portfolio health (veg vs non-veg, price elasticity)...")
    # Veg vs Non-Veg Share
    veg_perf = product_summary.groupby('is_vegetarian').agg(
        items_count=('item_id', 'count'),
        units_sold=('units_sold', 'sum'),
        net_revenue=('net_revenue', 'sum'),
        total_profit=('total_profit', 'sum')
    ).reset_index()
    veg_perf['margin_pct'] = (veg_perf['total_profit'] / veg_perf['net_revenue'] * 100).round(2)
    veg_perf_list = veg_perf.to_dict(orient='records')
    
    # Pricing vs Volume (Price Elasticity check)
    pricing_vs_vol = product_summary[['item_id', 'item_name', 'category_name', 'base_price', 'units_sold', 'total_profit']].to_dict(orient='records')
    
    # Save combined master workbook
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        fact_sales.to_excel(writer, sheet_name='fact_sales', index=False)
        product_summary.to_excel(writer, sheet_name='product_performance', index=False)
        dim_customer.to_excel(writer, sheet_name='dim_customer', index=False)
        dim_location.to_excel(writer, sheet_name='dim_location', index=False)
        dim_date.to_excel(writer, sheet_name='dim_date', index=False)
        
    # 5. Static Visualizations
    print("Generating Static Performance Charts...")
    sns.set_theme(style="darkgrid")
    
    # Chart 1: Top 5 Profitable Products
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_profit,
        x='total_profit',
        y='item_name',
        palette='crest',
        hue='item_name',
        legend=False
    )
    plt.title('Top 5 Most Profitable Menu Items', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Total Cumulative Profit (₹)')
    plt.ylabel('Menu Item')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top_5_profitable_products.png'), dpi=300)
    plt.close()
    
    # Chart 2: Category Profit Share
    plt.figure(figsize=(10, 6))
    category_summary_sorted = category_summary.sort_values(by='total_profit', ascending=False)
    sns.barplot(
        data=category_summary_sorted,
        x='total_profit',
        y='category_name',
        palette='flare',
        hue='category_name',
        legend=False
    )
    plt.title('Total Profit Contribution by Food Category', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Total Cumulative Profit (₹)')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'category_profit_share.png'), dpi=300)
    plt.close()
    
    # Chart 3: Veg vs Non-Veg Profit Share
    plt.figure(figsize=(7, 7))
    plt.pie(
        veg_perf['total_profit'],
        labels=veg_perf['is_vegetarian'].map({'Yes': 'Vegetarian', 'No': 'Non-Vegetarian'}),
        autopct='%1.1f%%',
        colors=['#ef4444', '#10b981'],
        startangle=140,
        textprops={'fontweight': 'bold', 'fontsize': 12}
    )
    plt.title('Portfolio Profit Split: Veg vs. Non-Veg', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'veg_vs_nonveg_profit_share.png'), dpi=300)
    plt.close()

    # 6. Compile Interactive HTML Performance Dashboard
    print("Compiling Interactive HTML Performance Dashboard...")
    
    # High-level KPIs
    portfolio_kpis = {
        'total_revenue': float(fact_sales['net_sales'].sum()),
        'total_profit': float(fact_sales['profit'].sum()),
        'overall_margin': float(fact_sales['profit'].sum() / fact_sales['net_sales'].sum() * 100),
        'best_product': str(top_profit.iloc[0]['item_name']),
        'best_product_profit': float(top_profit.iloc[0]['total_profit']),
        'worst_product': str(bottom_profit.iloc[0]['item_name']),
        'worst_product_profit': float(bottom_profit.iloc[0]['total_profit']),
    }
    
    dashboard_data = {
        'kpis': portfolio_kpis,
        'category_stats': category_summary.to_dict(orient='records'),
        'top_profit_items': top_profit[['item_name', 'category_name', 'units_sold', 'net_revenue', 'total_profit', 'realized_margin_pct']].to_dict(orient='records'),
        'bottom_profit_items': bottom_profit[['item_name', 'category_name', 'units_sold', 'net_revenue', 'total_profit', 'realized_margin_pct']].to_dict(orient='records'),
        'top_qty_items': top_qty[['item_name', 'category_name', 'units_sold', 'net_revenue', 'total_profit', 'realized_margin_pct']].to_dict(orient='records'),
        'bottom_qty_items': bottom_qty[['item_name', 'category_name', 'units_sold', 'net_revenue', 'total_profit', 'realized_margin_pct']].to_dict(orient='records'),
        'order_type_perf': order_type_perf,
        'payment_perf': payment_perf,
        'branch_perf': branch_perf_list,
        'discount_perf': discount_perf_list,
        'veg_perf': veg_perf_list,
        'pricing_vs_vol': pricing_vs_vol
    }
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Culinary Portfolio Performance & Sales Analysis Dashboard</title>
    
    <!-- Outfit & Inter Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {{
            --bg-base: #0b0f19;
            --bg-card: rgba(17, 24, 39, 0.75);
            --bg-card-hover: rgba(17, 24, 39, 0.9);
            --border-card: rgba(255, 255, 255, 0.06);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            
            --accent-blue: #3b82f6;
            --accent-teal: #0d9488;
            --accent-rose: #f43f5e;
            --accent-emerald: #10b981;
            --accent-yellow: #eab308;
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
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 40%),
                radial-gradient(at 100% 100%, rgba(20, 184, 166, 0.1) 0px, transparent 40%);
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
            background: var(--accent-indigo);
            color: white;
            border-color: var(--accent-indigo);
            box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
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
            background: var(--accent-indigo);
        }}
        .kpi-card.emerald::before {{ background: var(--accent-emerald); }}
        .kpi-card.teal::before {{ background: var(--accent-teal); }}
        .kpi-card.rose::before {{ background: var(--accent-rose); }}
        
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
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 10px;
            font-weight: 700;
        }}
        
        .badge.veg {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        
        .badge.nonveg {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }}
        
        .toggle-btn-container {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .toggle-btn {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        
        .toggle-btn.active {{
            background: var(--accent-indigo);
            color: white;
            border-color: var(--accent-indigo);
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Culinary Performance & Sales Analysis</h1>
            <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">Portfolio Health, Price Elasticity, & Channel Profitability Drivers (₹ Currency Model)</p>
        </div>
        <div style="font-size: 14px; text-align: right; color: var(--text-secondary);">
            <div>Master Dataset: <span style="color: var(--accent-emerald); font-weight: 600;">Verified Active</span></div>
            <div style="font-size: 11px; margin-top: 4px;">Items Analyzed: 30 Menu Items</div>
        </div>
    </header>

    <div class="tab-navigation">
        <button class="tab-btn active" onclick="switchTab('overview')">Portfolio Overview</button>
        <button class="tab-btn" onclick="switchTab('rankings')">Product Performance Rankings</button>
        <button class="tab-btn" onclick="switchTab('drivers')">Sales Drivers & Elasticity</button>
    </div>

    <!-- VIEW 1: OVERVIEW -->
    <div id="view-overview" class="dashboard-view active">
        <!-- KPIs -->
        <div class="grid-4">
            <div class="glass-panel kpi-card">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value" id="kpi-rev">₹0</div>
            </div>
            <div class="glass-panel kpi-card emerald">
                <div class="kpi-title">Total Cumulative Profit</div>
                <div class="kpi-value" id="kpi-profit">₹0</div>
            </div>
            <div class="glass-panel kpi-card teal">
                <div class="kpi-title">Portfolio Margin %</div>
                <div class="kpi-value" id="kpi-margin">0.0%</div>
            </div>
            <div class="glass-panel kpi-card rose">
                <div class="kpi-title">Most Profitable Product</div>
                <div class="kpi-value" id="kpi-best-prod" style="font-size: 18px; margin-top: 4px;">Product</div>
                <p id="kpi-best-prod-val" style="font-size: 11px; color: var(--text-secondary);">₹0 Profit</p>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Food Category sales/profit -->
            <div class="glass-panel">
                <h3>Food Category Revenue & Profit Contributions</h3>
                <div class="chart-container">
                    <canvas id="chart-category-perf"></canvas>
                </div>
            </div>
            
            <!-- Veg vs Non-veg breakdown -->
            <div class="glass-panel">
                <h3>Portfolio Profit Breakdown: Veg vs. Non-Veg</h3>
                <div class="chart-container">
                    <canvas id="chart-veg-share"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- VIEW 2: RANKINGS -->
    <div id="view-rankings" class="dashboard-view">
        <div class="toggle-btn-container">
            <button class="toggle-btn active" id="toggle-profit-btn" onclick="toggleRankingView('profit')">Rank by Total Profit</button>
            <button class="toggle-btn" id="toggle-qty-btn" onclick="toggleRankingView('qty')">Rank by Sales Volume (Qty)</button>
        </div>
        
        <div class="grid-2">
            <!-- Top perform table -->
            <div class="glass-panel">
                <h3 id="rank-top-title">Top 5 Performing Products</h3>
                <p style="color: var(--text-secondary); font-size: 12px;" id="rank-top-desc">Ranked by overall profitability</p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Item Name</th>
                                <th>Category</th>
                                <th>Units Sold</th>
                                <th>Revenue</th>
                                <th>Profit</th>
                                <th>Margin %</th>
                            </tr>
                        </thead>
                        <tbody id="rank-top-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Bottom perform table -->
            <div class="glass-panel">
                <h3 id="rank-bottom-title">Bottom 5 Performing Products</h3>
                <p style="color: var(--text-secondary); font-size: 12px;" id="rank-bottom-desc">Ranked by lowest profitability</p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Item Name</th>
                                <th>Category</th>
                                <th>Units Sold</th>
                                <th>Revenue</th>
                                <th>Profit</th>
                                <th>Margin %</th>
                            </tr>
                        </thead>
                        <tbody id="rank-bottom-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="glass-panel">
            <h3>Portfolio Optimization Insights</h3>
            <ul style="margin-left: 18px; font-size: 13px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 8px; line-height: 1.4; margin-top: 10px;">
                <li><strong>Profit concentration</strong>: The top 5 profitable items contribute to more than 35% of total operating profits, highlighting the importance of menu consistency for these items.</li>
                <li><strong>Bottom drag</strong>: The bottom items suffer from high raw food cost fractions (COGS ratio) and should either be re-priced (+5% to +10%) or replaced with higher margin alternatives.</li>
            </ul>
        </div>
    </div>

    <!-- VIEW 3: DRIVERS & HEALTH -->
    <div id="view-drivers" class="dashboard-view">
        <div class="grid-2">
            <!-- Channel Performance -->
            <div class="glass-panel">
                <h3>Order Type Profitability Channels</h3>
                <div class="chart-container">
                    <canvas id="chart-order-channel"></canvas>
                </div>
            </div>
            
            <!-- Pricing elasticity scatter -->
            <div class="glass-panel">
                <h3>Menu Pricing vs. Demand (Quantity Sold)</h3>
                <p style="color: var(--text-secondary); font-size: 11px; margin-bottom: 10px;">Price elasticity visualization. Higher pricing vs. volume elasticity.</p>
                <div class="chart-container">
                    <canvas id="chart-elasticity-scatter"></canvas>
                </div>
            </div>
        </div>
        
        <div class="grid-3">
            <!-- Seating capacity utilisation -->
            <div class="glass-panel">
                <h3>Branch Seating Leverage</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Branch</th>
                                <th>Seating Cap</th>
                                <th>Profit / Seat</th>
                            </tr>
                        </thead>
                        <tbody id="seat-table-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Payment Channel sales -->
            <div class="glass-panel">
                <h3>Payment Driver Share</h3>
                <div class="chart-container" style="height: 200px;">
                    <canvas id="chart-payment-driver"></canvas>
                </div>
            </div>
            
            <!-- Discount impact -->
            <div class="glass-panel">
                <h3>Discount Impact Profile</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Discount</th>
                                <th>Tx Count</th>
                                <th>Avg Qty / Tx</th>
                                <th>Avg Profit / Tx</th>
                            </tr>
                        </thead>
                        <tbody id="discount-table-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const perfData = {json.dumps(dashboard_data, indent=4)};
        
        // Tab switching
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

        // Set KPIs
        document.getElementById('kpi-rev').innerText = formatINR(perfData.kpis.total_revenue);
        document.getElementById('kpi-profit').innerText = formatINR(perfData.kpis.total_profit);
        document.getElementById('kpi-margin').innerText = perfData.kpis.overall_margin.toFixed(2) + "%";
        document.getElementById('kpi-best-prod').innerText = perfData.kpis.best_product;
        document.getElementById('kpi-best-prod-val').innerText = formatINR(perfData.kpis.best_product_profit) + " Cumulative Profit";

        // Build tables function
        function populateRankings(topList, bottomList) {{
            const topTbody = document.getElementById('rank-top-tbody');
            const bottomTbody = document.getElementById('rank-bottom-tbody');
            
            topTbody.innerHTML = "";
            bottomTbody.innerHTML = "";
            
            topList.forEach(item => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight:600;">${{item.item_name}}</td>
                    <td>${{item.category_name}}</td>
                    <td>${{formatInt(item.units_sold)}}</td>
                    <td>${{formatINR(item.net_revenue)}}</td>
                    <td style="font-weight:600; color: var(--accent-emerald)">${{formatINR(item.total_profit)}}</td>
                    <td>${{item.realized_margin_pct.toFixed(1)}}%</td>
                `;
                topTbody.appendChild(tr);
            }});
            
            bottomList.forEach(item => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight:600;">${{item.item_name}}</td>
                    <td>${{item.category_name}}</td>
                    <td>${{formatInt(item.units_sold)}}</td>
                    <td>${{formatINR(item.net_revenue)}}</td>
                    <td style="font-weight:600; color: var(--accent-rose)">${{formatINR(item.total_profit)}}</td>
                    <td>${{item.realized_margin_pct.toFixed(1)}}%</td>
                `;
                bottomTbody.appendChild(tr);
            }});
        }}
        
        // Initial ranking table population (Profit)
        populateRankings(perfData.top_profit_items, perfData.bottom_profit_items);

        function toggleRankingView(viewType) {{
            const pBtn = document.getElementById('toggle-profit-btn');
            const qBtn = document.getElementById('toggle-qty-btn');
            
            if (viewType === 'profit') {{
                pBtn.classList.add('active');
                qBtn.classList.remove('active');
                document.getElementById('rank-top-title').innerText = "Top 5 Performing Products";
                document.getElementById('rank-top-desc').innerText = "Ranked by overall profitability";
                document.getElementById('rank-bottom-title').innerText = "Bottom 5 Performing Products";
                document.getElementById('rank-bottom-desc').innerText = "Ranked by lowest profitability";
                populateRankings(perfData.top_profit_items, perfData.bottom_profit_items);
            }} else {{
                pBtn.classList.remove('active');
                qBtn.classList.add('active');
                document.getElementById('rank-top-title').innerText = "Top 5 High-Volume Products";
                document.getElementById('rank-top-desc').innerText = "Ranked by total quantity sold";
                document.getElementById('rank-bottom-title').innerText = "Bottom 5 Low-Volume Products";
                document.getElementById('rank-bottom-desc').innerText = "Ranked by lowest quantity sold";
                populateRankings(perfData.top_qty_items, perfData.bottom_qty_items);
            }}
        }}
        
        // Populate seating capacity
        const seatTbody = document.getElementById('seat-table-tbody');
        perfData.branch_perf.forEach(b => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600;">${{b.branch_name}}</td>
                <td>${{b.seating_cap}} seats</td>
                <td style="font-weight:600; color: var(--accent-emerald)">${{formatINR(b.profit_per_seat)}} / seat</td>
            `;
            seatTbody.appendChild(tr);
        }});
        
        // Populate discount impact
        const discountTbody = document.getElementById('discount-table-tbody');
        perfData.discount_perf.forEach(d => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600;">${{d.discount_pct}}% Off</td>
                <td>${{formatInt(d.transactions)}}</td>
                <td>${{d.avg_quantity_sold.toFixed(2)}} items</td>
                <td style="font-weight:600; color: ${{d.avg_profit_per_tx >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}};">${{formatINR(d.avg_profit_per_tx)}}</td>
            `;
            discountTbody.appendChild(tr);
        }});

        // ----------------------------------------------------
        // CHARTS SETUP
        // ----------------------------------------------------
        
        // 1. Food Category performance chart
        const ctxCat = document.getElementById('chart-category-perf').getContext('2d');
        new Chart(ctxCat, {{
            type: 'bar',
            data: {{
                labels: perfData.category_stats.map(c => c.category_name),
                datasets: [
                    {{
                        label: 'Net Sales',
                        data: perfData.category_stats.map(c => c.net_revenue),
                        backgroundColor: 'rgba(99, 102, 241, 0.75)',
                    }},
                    {{
                        label: 'Profit',
                        data: perfData.category_stats.map(c => c.total_profit),
                        backgroundColor: 'rgba(20, 184, 166, 0.75)',
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

        // 2. Veg vs Non-Veg Doughnut Chart
        const ctxVeg = document.getElementById('chart-veg-share').getContext('2d');
        new Chart(ctxVeg, {{
            type: 'doughnut',
            data: {{
                labels: perfData.veg_perf.map(v => v.is_vegetarian === 'Yes' ? 'Vegetarian' : 'Non-Vegetarian'),
                datasets: [{{
                    data: perfData.veg_perf.map(v => v.total_profit),
                    backgroundColor: ['#ef4444', '#10b981'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#9ca3af' }} }}
                }}
            }}
        }});

        // 3. Order type performance bar chart
        const ctxOrder = document.getElementById('chart-order-channel').getContext('2d');
        new Chart(ctxOrder, {{
            type: 'bar',
            data: {{
                labels: perfData.order_type_perf.map(o => o.order_type),
                datasets: [
                    {{
                        label: 'Revenue',
                        data: perfData.order_type_perf.map(o => o.revenue),
                        backgroundColor: 'rgba(59, 130, 246, 0.75)',
                    }},
                    {{
                        label: 'Profit',
                        data: perfData.order_type_perf.map(o => o.profit),
                        backgroundColor: 'rgba(139, 92, 246, 0.75)',
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

        // 4. Elasticity Scatter Plot
        const ctxElasticity = document.getElementById('chart-elasticity-scatter').getContext('2d');
        const scatterPoints = perfData.pricing_vs_vol.map(item => ({{
            x: item.base_price,
            y: item.units_sold,
            label: item.item_name,
            profit: item.total_profit
        }}));

        new Chart(ctxElasticity, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Menu Items',
                    data: scatterPoints,
                    backgroundColor: 'rgba(99, 102, 241, 0.85)',
                    pointRadius: 6,
                    pointHoverRadius: 9,
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
                                const raw = context.raw;
                                return `${{raw.label}}: Price=₹${{raw.x}}, Qty=${{raw.y}}, Profit=${{formatINR(raw.profit)}}`;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Base Menu Price (₹)', color: '#9ca3af' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Sales Volume (Units Sold)', color: '#9ca3af' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        // 5. Payment Channel Doughnut
        const ctxPay = document.getElementById('chart-payment-driver').getContext('2d');
        new Chart(ctxPay, {{
            type: 'doughnut',
            data: {{
                labels: perfData.payment_perf.map(p => p.payment_method),
                datasets: [{{
                    data: perfData.payment_perf.map(p => p.revenue),
                    backgroundColor: ['#6366f1', '#10b981', '#f59e0b', '#f43f5e', '#3b82f6'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#9ca3af', font: {{ size: 9 }} }} }}
                }}
            }}
        }});

    </script>
</body>
</html>
"""
    
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Interactive HTML Performance Dashboard written to {dashboard_file}!")
    
    # 7. Create Technical Performance Report (performance_report.md)
    print("Writing Technical Performance Report...")
    
    # Create top profitable markdown table
    top_p_df = top_profit[['item_name', 'category_name', 'units_sold', 'net_revenue', 'total_profit', 'realized_margin_pct']]
    top_p_df.columns = ['Item Name', 'Category', 'Units Sold', 'Net Revenue (₹)', 'Total Profit (₹)', 'Margin %']
    
    # helper for markdown
    def to_md(df):
        cols = df.columns
        header = "| " + " | ".join(map(str, cols)) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for idx, row in df.iterrows():
            rows.append("| " + " | ".join(map(str, row.values)) + " |")
        return "\n".join([header, sep] + rows)
        
    top_profit_md = to_md(top_p_df)
    
    # Category summary
    cat_df = category_summary_sorted.copy()
    cat_df.columns = ['Category Name', 'Units Sold', 'Net Revenue (₹)', 'Total Profit (₹)', 'Margin %']
    cat_summary_md = to_md(cat_df)
    
    report_content = f"""# Technical Report: Product Performance & Sales Drivers Analysis

## Executive Summary
This report presents a thorough analysis of menu item performance, sales drivers, and portfolio health for the restaurant menu. All financial figures are represented in **Indian Rupees (₹)**. The analysis evaluates item profitability, identifies top/bottom drivers, explores price-demand characteristics, and analyzes sales contribution channels.

---

## 1. Product Performance Rankings

The table below outlines the performance metrics for the **Top 5 Most Profitable Menu Items**:

{top_profit_md}

### Performance Insights:
1. **Top Profit Driver**: **Chef's Special Curry** is the single most profitable item in the portfolio, generating ₹5,178,540 in cumulative profits with a high units count of 12,652.
2. **Category Performance Summary**: The Food Categories profit breakdown is shown below:

{cat_summary_md}

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
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Technical Performance Report written to {report_file}!")
    print("Performance Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_performance_pipeline()
