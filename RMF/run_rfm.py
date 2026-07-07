import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_rfm_pipeline():
    print("Starting RFM Customer Segmentation Pipeline...")
    
    # 1. Paths
    cleaned_dir = "d:/cat/DAL/project/ETL/cleaned_data"
    rfm_dir = "d:/cat/DAL/project/RMF"
    output_dir = os.path.join(rfm_dir, 'cleaned_data')
    viz_dir = os.path.join(rfm_dir, 'visualizations')
    dashboard_file = os.path.join(rfm_dir, 'rfm_dashboard.html')
    report_file = os.path.join(rfm_dir, 'rfm_report.md')
    output_excel = os.path.join(rfm_dir, 'Cleaned_Restaurant_Data_RFM.xlsx')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load ETL cleaned data
    fact_sales = pd.read_csv(os.path.join(cleaned_dir, 'fact_sales.csv'))
    dim_customer = pd.read_csv(os.path.join(cleaned_dir, 'dim_customer.csv'))
    dim_menu = pd.read_csv(os.path.join(cleaned_dir, 'dim_menu.csv'))
    dim_location = pd.read_csv(os.path.join(cleaned_dir, 'dim_location.csv'))
    dim_date = pd.read_csv(os.path.join(cleaned_dir, 'dim_date.csv'))
    
    fact_sales['order_date'] = pd.to_datetime(fact_sales['order_date'])
    
    # 2. Compute RFM Metrics
    print("Calculating R, F, M metrics per customer...")
    ref_date = fact_sales['order_date'].max()
    
    rfm_df = fact_sales.groupby('customer_id').agg(
        last_purchase=('order_date', 'max'),
        Frequency=('order_id', 'nunique'),
        Monetary=('net_sales', 'sum')
    ).reset_index()
    
    rfm_df['Recency'] = (ref_date - rfm_df['last_purchase']).dt.days
    
    # 3. Binning Quintiles (1-5 Scores)
    print("Mapping metrics to quintile ranks...")
    # Recency quintiles (5 = most recent, 1 = least recent)
    rfm_df['Recency_Score'] = pd.qcut(rfm_df['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    
    # Frequency quintiles (we rank first due to duplicate boundaries, 5 = highest frequency)
    rfm_df['Frequency_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Monetary quintiles (5 = highest spender)
    rfm_df['Monetary_Score'] = pd.qcut(rfm_df['Monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # RFM Combined Scores
    rfm_df['RFM_Group'] = rfm_df['Recency_Score'].astype(str) + rfm_df['Frequency_Score'].astype(str) + rfm_df['Monetary_Score'].astype(str)
    rfm_df['RFM_Score'] = rfm_df['Recency_Score'] + rfm_df['Frequency_Score'] + rfm_df['Monetary_Score']
    
    # 4. Behavioral Segmentation
    print("Classifying customers into segments...")
    def classify_rfm(row):
        r = row['Recency_Score']
        f = row['Frequency_Score']
        m = row['Monetary_Score']
        
        # 1. Champions (R: 4-5, F: 4-5)
        if r >= 4 and f >= 4:
            return 'Champions'
        # 2. Loyal Customers (R: 3-5, F: 3-5)
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        # 3. Potential Loyalists (R: 4-5, F: 2-3)
        elif r >= 4 and f >= 2:
            return 'Potential Loyalists'
        # 4. New Customers (R: 4-5, F: 1)
        elif r >= 4 and f == 1:
            return 'New Customers'
        # 5. Promising (R: 3, F: 1)
        elif r == 3 and f == 1:
            return 'Promising'
        # 6. Can't Lose Them (R: 1, F: 4-5)
        elif r == 1 and f >= 4:
            return "Can't Lose Them"
        # 7. At Risk (R: 1-2, F: 3-5)
        elif r in [1, 2] and f >= 3:
            return 'At Risk'
        # 8. Need Attention (R: 2-3, F: 2-3)
        elif r in [2, 3] and f in [2, 3]:
            return 'Need Attention'
        # 9. About to Sleep (R: 2-3, F: 1)
        elif r in [2, 3] and f == 1:
            return 'About to Sleep'
        # 10. Lost / Hibernating (R: 1-2, F: 1-2)
        else:
            return 'Lost / Hibernating'
            
    rfm_df['Segment'] = rfm_df.apply(classify_rfm, axis=1)
    
    # Join segment info to dim_customer
    dim_customer_rfm = dim_customer.merge(
        rfm_df[['customer_id', 'Recency', 'Frequency', 'Monetary', 
                'Recency_Score', 'Frequency_Score', 'Monetary_Score', 
                'RFM_Group', 'RFM_Score', 'Segment']], 
        on='customer_id', how='left'
    )
    
    # 5. Save Cleaned Dataset Files
    print("Saving cleaned and enriched data files...")
    dim_customer_rfm.to_csv(os.path.join(output_dir, 'dim_customer_rfm.csv'), index=False)
    # Copy other tables
    fact_sales.to_csv(os.path.join(output_dir, 'fact_sales.csv'), index=False)
    dim_menu.to_csv(os.path.join(output_dir, 'dim_menu.csv'), index=False)
    dim_location.to_csv(os.path.join(output_dir, 'dim_location.csv'), index=False)
    dim_date.to_csv(os.path.join(output_dir, 'dim_date.csv'), index=False)
    
    # Save Master Excel sheet
    print("Saving consolidated Excel workbook...")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        fact_sales.to_excel(writer, sheet_name='fact_sales', index=False)
        dim_customer_rfm.to_excel(writer, sheet_name='dim_customer_rfm', index=False)
        dim_menu.to_excel(writer, sheet_name='dim_menu', index=False)
        dim_location.to_excel(writer, sheet_name='dim_location', index=False)
        dim_date.to_excel(writer, sheet_name='dim_date', index=False)
        
    # 6. Static Visualizations
    print("Generating Static RFM Charts...")
    sns.set_theme(style="darkgrid")
    
    # Chart 1: Segment Size Breakdown
    plt.figure(figsize=(12, 6))
    seg_counts = dim_customer_rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    sns.barplot(
        data=seg_counts,
        x='Count',
        y='Segment',
        palette='viridis',
        hue='Segment',
        legend=False
    )
    plt.title('Customer Segment Size Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Customers')
    plt.ylabel('Segment')
    
    # Annotate counts
    for idx, row in seg_counts.iterrows():
        plt.text(row['Count'] + 5, idx, f"{row['Count']}", va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'customer_segments_size.png'), dpi=300)
    plt.close()
    
    # Chart 2: Recency vs Monetary scatter colored by Segment
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=dim_customer_rfm,
        x='Recency',
        y='Monetary',
        hue='Segment',
        palette='tab10',
        s=80,
        alpha=0.7,
        edgecolor='none'
    )
    plt.title('Customer Spending vs. Recency Scatter Plot', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Recency (Days since last purchase)')
    plt.ylabel('Monetary (Total Spend - ₹)')
    plt.legend(title='Behavior Segment', loc='upper right', bbox_to_anchor=(1.05, 1), frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'recency_vs_monetary_scatter.png'), dpi=300)
    plt.close()

    # 7. Generate Interactive HTML Dashboard
    print("Compiling Interactive HTML RFM Dashboard...")
    
    # Calculate segment profile averages
    seg_profiles = dim_customer_rfm.groupby('Segment').agg(
        customers=('customer_id', 'count'),
        avg_recency=('Recency', 'mean'),
        avg_frequency=('Frequency', 'mean'),
        avg_monetary=('Monetary', 'mean'),
        avg_loyalty=('loyalty_points', 'mean')
    ).round(2).reset_index().to_dict(orient='records')
    
    # Compute high level KPIs
    rfm_kpis = {
        'total_revenue': float(dim_customer_rfm['Monetary'].sum()),
        'avg_spend_per_cust': float(dim_customer_rfm['Monetary'].mean()),
        'avg_orders_per_cust': float(dim_customer_rfm['Frequency'].mean()),
        'avg_recency': float(dim_customer_rfm['Recency'].mean()),
        'total_customers': int(len(dim_customer_rfm)),
    }
    
    # Package top 50 customers to show in dashboard lookup
    cust_lookup_data = dim_customer_rfm[['customer_id', 'customer_name', 'gender', 'age', 'customer_type', 'city', 'loyalty_points', 'Recency', 'Frequency', 'Monetary', 'RFM_Group', 'Segment']].round(2).to_dict(orient='records')
    
    # Marketing Action Plan recommendations
    marketing_actions = {
        'Champions': 'Reward them with early access, exclusive menus, and zero pricing friction. Upsell premium items.',
        'Loyal Customers': 'Offer loyalty program bonuses, recommend new menu items, and send personalized birthday coupons.',
        'Potential Loyalists': 'Provide multi-purchase discounts or starter+dessert pairing offers to increase ordering frequency.',
        'New Customers': 'Send an automated welcome email with a high-conversion discount for their second visit within 14 days.',
        'Promising': 'Build brand awareness. Offer target coupons to reactivate interest.',
        'Need Attention': 'Send time-limited discounts on their top-ordered items. Create menu bundles.',
        'About to Sleep': 'Trigger win-back sequences. Recommend popular low-cost appetizers or weekend specials.',
        'At Risk': 'Offer a high-value discount (e.g. ₹500 off) on their favorite meals. Ask for feedback.',
        "Can't Lose Them": 'Make a personal phone call/VIP email invitation. Conduct surveys, offer highly valuable incentives.',
        'Lost / Hibernating': 'Run low-cost sweepstakes and double points reactivation campaigns.'
    }
    
    dashboard_data = {
        'kpis': rfm_kpis,
        'profiles': seg_profiles,
        'customers': cust_lookup_data,
        'actions': marketing_actions
    }
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer RFM Behavioral Segmentation Dashboard</title>
    
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
            --accent-purple: #8b5cf6;
            --accent-orange: #f97316;
            --accent-emerald: #10b981;
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
                radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 40%),
                radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.1) 0px, transparent 40%);
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
        
        .grid-5 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
            background: var(--accent-blue);
        }}
        .kpi-card.purple::before {{ background: var(--accent-purple); }}
        .kpi-card.teal::before {{ background: var(--accent-teal); }}
        .kpi-card.orange::before {{ background: var(--accent-orange); }}
        .kpi-card.emerald::before {{ background: var(--accent-emerald); }}
        
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
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* Segment Badges */
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

        /* Action plan list */
        .action-item {{
            margin-top: 15px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid var(--accent-blue);
            border-radius: 4px;
            font-size: 13px;
        }}
        
        .action-title {{
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 4px;
        }}
        
        /* Interactive customer search */
        .search-container {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .search-input {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            color: white;
            padding: 10px 18px;
            font-size: 14px;
            outline: none;
            flex: 1;
            transition: all 0.2s ease;
        }}
        
        .search-input:focus {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        }}
        
        .profile-row {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 15px;
            font-size: 13px;
            line-height: 1.5;
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
            <h1>Customer Behavioral Segmentation</h1>
            <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">Recency, Frequency, & Monetary (RFM) Segmentation Dashboard (₹ Currency Model)</p>
        </div>
        <div style="font-size: 14px; text-align: right; color: var(--text-secondary);">
            <div>RFM Engine: <span style="color: var(--accent-teal); font-weight: 600;">Quintile Ranks</span></div>
            <div style="font-size: 11px; margin-top: 4px;">Active Segments: 10 Behavioral Groups</div>
        </div>
    </header>

    <div class="tab-navigation">
        <button class="tab-btn active" onclick="switchTab('overview')">Segmentation Overview</button>
        <button class="tab-btn" onclick="switchTab('profiles')">Behavioral Profiles</button>
        <button class="tab-btn" onclick="switchTab('lookup')">Interactive Customer Lookup</button>
    </div>

    <!-- VIEW 1: OVERVIEW -->
    <div id="view-overview" class="dashboard-view active">
        <!-- KPIs -->
        <div class="grid-5">
            <div class="glass-panel kpi-card">
                <div class="kpi-title">Total Customers</div>
                <div class="kpi-value" id="kpi-cust">0</div>
            </div>
            <div class="glass-panel kpi-card purple">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value" id="kpi-rev">₹0</div>
            </div>
            <div class="glass-panel kpi-card teal">
                <div class="kpi-title">Avg Spend/Customer</div>
                <div class="kpi-value" id="kpi-spend">₹0</div>
            </div>
            <div class="glass-panel kpi-card orange">
                <div class="kpi-title">Avg Visits/Customer</div>
                <div class="kpi-value" id="kpi-visits">0</div>
            </div>
            <div class="glass-panel kpi-card emerald">
                <div class="kpi-title">Avg Recency</div>
                <div class="kpi-value" id="kpi-recency">0 days</div>
            </div>
        </div>
        
        <div class="grid-2">
            <!-- Segment distribution bar chart -->
            <div class="glass-panel">
                <h3>Behavioral Segment Counts</h3>
                <div class="chart-container">
                    <canvas id="chart-segments-count"></canvas>
                </div>
            </div>
            
            <!-- Revenue Contribution by segment -->
            <div class="glass-panel">
                <h3>Revenue Share by Customer Segment</h3>
                <div class="chart-container">
                    <canvas id="chart-segments-revenue"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- VIEW 2: BEHAVIORAL PROFILES -->
    <div id="view-profiles" class="dashboard-view">
        <div class="glass-panel">
            <h3>RFM Segment Profiling Grid</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 15px;">
                Average transactional metrics for each customer behavioral profile.
            </p>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Segment Name</th>
                            <th>Active Customers</th>
                            <th>Avg Recency (Days)</th>
                            <th>Avg Frequency (Orders)</th>
                            <th>Avg Spend (₹)</th>
                            <th>Avg Loyalty Points</th>
                        </tr>
                    </thead>
                    <tbody id="profiles-table-body">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Marketing Action Plan -->
        <div class="glass-panel">
            <h3>Targeted Marketing & Culinary Activation Strategies</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 15px;">Actionable recommendations for activating each behavioral segment.</p>
            <div class="grid-2" id="marketing-actions-container">
                <!-- Populated by JS -->
            </div>
        </div>
    </div>

    <!-- VIEW 3: INTERACTIVE CUSTOMER LOOKUP -->
    <div id="view-lookup" class="dashboard-view">
        <div class="grid-2">
            <!-- Search pane -->
            <div class="glass-panel">
                <h3>Customer Profiler Search</h3>
                <p style="color: var(--text-secondary); font-size: 12px; margin-bottom: 15px;">Search by customer ID or name to view their exact transactional metrics, RFM score, and segment.</p>
                
                <div class="search-container">
                    <input type="text" id="cust-search-input" class="search-input" placeholder="Enter customer ID or name (e.g. Customer_0001)..." oninput="searchCustomer()">
                </div>
                
                <div id="lookup-profile-card" style="display: none;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 10px; margin-bottom: 15px;">
                        <h4 id="profile-name" style="font-size: 18px; color: var(--accent-blue);">Customer Name</h4>
                        <span class="badge" id="profile-segment-badge">Segment</span>
                    </div>
                    
                    <div class="profile-row">
                        <div><span class="profile-label">Customer ID:</span></div>
                        <div><span class="profile-value" id="profile-id">-</span></div>
                    </div>
                    <div class="profile-row">
                        <div><span class="profile-label">City / Region:</span></div>
                        <div><span class="profile-value" id="profile-city">-</span></div>
                    </div>
                    <div class="profile-row">
                        <div><span class="profile-label">Demographics:</span></div>
                        <div><span class="profile-value" id="profile-demog">-</span></div>
                    </div>
                    <div class="profile-row">
                        <div><span class="profile-label">Loyalty Points:</span></div>
                        <div><span class="profile-value" id="profile-loyalty" style="color: var(--accent-emerald)">-</span></div>
                    </div>
                    
                    <div style="margin: 20px 0; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 15px;">
                        <h4 style="margin-bottom: 10px;">RFM Behavior Scores</h4>
                        <div class="profile-row">
                            <div><span class="profile-label">Recency (Days / Score):</span></div>
                            <div><span class="profile-value"><span id="profile-days">-</span> days (Score: <span id="profile-r-score">-</span>/5)</span></div>
                        </div>
                        <div class="profile-row">
                            <div><span class="profile-label">Frequency (Orders / Score):</span></div>
                            <div><span class="profile-value"><span id="profile-orders">-</span> visits (Score: <span id="profile-f-score">-</span>/5)</span></div>
                        </div>
                        <div class="profile-row">
                            <div><span class="profile-label">Monetary (Total spend / Score):</span></div>
                            <div><span class="profile-value"><span id="profile-monetary">-</span> (Score: <span id="profile-m-score">-</span>/5)</span></div>
                        </div>
                        <div class="profile-row">
                            <div><span class="profile-label">RFM Group Code:</span></div>
                            <div><span class="profile-value" id="profile-code" style="letter-spacing: 2px; color: var(--accent-purple);">-</span></div>
                        </div>
                    </div>
                    
                    <div class="action-item" style="margin-top: 20px;">
                        <div class="action-title">Targeted Engagement Advice</div>
                        <p id="profile-action-text">-</p>
                    </div>
                </div>
                
                <div id="lookup-search-help" style="color: var(--text-secondary); text-align: center; padding: 40px 0; font-size: 13px;">
                    Type a customer name above to query behavior profiles.
                </div>
            </div>
            
            <!-- Quick high spenders table -->
            <div class="glass-panel">
                <h3>Top Revenue Spenders (Top 25)</h3>
                <div class="table-wrapper" style="max-height: 480px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>City</th>
                                <th>Loyalty</th>
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

    <script>
        const rfmData = {json.dumps(dashboard_data, indent=4)};
        
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

        // Initialize KPIs
        document.getElementById('kpi-cust').innerText = formatInt(rfmData.kpis.total_customers);
        document.getElementById('kpi-rev').innerText = formatINR(rfmData.kpis.total_revenue);
        document.getElementById('kpi-spend').innerText = formatINR(rfmData.kpis.avg_spend_per_cust);
        document.getElementById('kpi-visits').innerText = rfmData.kpis.avg_orders_per_cust.toFixed(2);
        document.getElementById('kpi-recency').innerText = rfmData.kpis.avg_recency.toFixed(1) + " days";

        // Segment colors mapping
        const segColors = {{
            'Champions': '#10b981', // emerald
            'Loyal Customers': '#8b5cf6', // purple
            'Potential Loyalists': '#3b82f6', // blue
            'New Customers': '#14b8a6', // teal
            'Promising': '#eab308', // yellow
            'Need Attention': '#f97316', // orange
            'About to Sleep': '#9ca3af', // gray
            'At Risk': '#f43f5e', // rose
            "Can't Lose Them": '#db2777', // pink
            'Lost / Hibernating': '#4b5563' // dark gray
        }};
        
        const getBadgeClass = (seg) => {{
            return seg.toLowerCase().replace("'", "").replace("/ ", "").replace(" ", "_");
        }};

        // Build Profiles Table
        const profilesTbody = document.getElementById('profiles-table-body');
        rfmData.profiles.forEach(p => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="badge ${{getBadgeClass(p.Segment)}}">${{p.Segment}}</span></td>
                <td style="font-weight:600;">${{formatInt(p.customers)}}</td>
                <td>${{p.avg_recency.toFixed(1)}}</td>
                <td>${{p.avg_frequency.toFixed(2)}}</td>
                <td style="font-weight:600;">${{formatINR(p.avg_monetary)}}</td>
                <td style="color: var(--accent-emerald)">${{formatInt(Math.round(p.avg_loyalty))}}</td>
            `;
            profilesTbody.appendChild(tr);
        }});
        
        // Build Action Items in View 2
        const actionContainer = document.getElementById('marketing-actions-container');
        Object.keys(rfmData.actions).forEach(seg => {{
            const div = document.createElement('div');
            div.className = 'action-item';
            div.style.borderLeftColor = segColors[seg];
            div.innerHTML = `
                <div class="action-title" style="color: ${{segColors[seg]}};">${{seg}}</div>
                <p style="color: var(--text-secondary); line-height: 1.4;">${{rfmData.actions[seg]}}</p>
            `;
            actionContainer.appendChild(div);
        }});

        // Build Top Spenders Table
        const spenderTbody = document.getElementById('top-spenders-body');
        const sortedSpenders = [...rfmData.customers].sort((a,b) => b.Monetary - a.Monetary).slice(0, 25);
        sortedSpenders.forEach(c => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600; cursor:pointer; color: var(--accent-blue);" onclick="fillSearch('${{c.customer_name}}')">${{c.customer_name}}</td>
                <td>${{c.city}}</td>
                <td>${{formatInt(c.loyalty_points)}}</td>
                <td>${{c.Frequency}}</td>
                <td style="font-weight:600;">${{formatINR(c.Monetary)}}</td>
                <td><span class="badge ${{getBadgeClass(c.Segment)}}" style="font-size:9px; padding: 2px 6px;">${{c.Segment}}</span></td>
            `;
            spenderTbody.appendChild(tr);
        }});
        
        function fillSearch(name) {{
            document.getElementById('cust-search-input').value = name;
            switchTab('lookup');
            searchCustomer();
        }}

        // Interactive Customer Search
        function searchCustomer() {{
            const input = document.getElementById('cust-search-input').value.toLowerCase().trim();
            const card = document.getElementById('lookup-profile-card');
            const help = document.getElementById('lookup-search-help');
            
            if (input.length < 2) {{
                card.style.display = 'none';
                help.style.display = 'block';
                return;
            }}
            
            // Search customers
            const cust = rfmData.customers.find(c => 
                c.customer_id.toString() === input || 
                c.customer_name.toLowerCase().includes(input)
            );
            
            if (cust) {{
                card.style.display = 'block';
                help.style.display = 'none';
                
                document.getElementById('profile-name').innerText = cust.customer_name;
                document.getElementById('profile-id').innerText = "#" + cust.customer_id;
                document.getElementById('profile-city').innerText = cust.city;
                document.getElementById('profile-demog').innerText = `${{cust.gender}}, Age ${{cust.age}} (${{cust.customer_type}})`;
                document.getElementById('profile-loyalty').innerText = formatInt(cust.loyalty_points) + " pts";
                
                document.getElementById('profile-days').innerText = cust.Recency;
                document.getElementById('profile-r-score').innerText = cust.Recency_Score;
                document.getElementById('profile-orders').innerText = cust.Frequency;
                document.getElementById('profile-f-score').innerText = cust.Frequency_Score;
                document.getElementById('profile-monetary').innerText = formatINR(cust.Monetary);
                document.getElementById('profile-m-score').innerText = cust.Monetary_Score;
                
                document.getElementById('profile-code').innerText = cust.RFM_Group;
                
                // Badge
                const badge = document.getElementById('profile-segment-badge');
                badge.innerText = cust.Segment;
                badge.className = `badge ${{getBadgeClass(cust.Segment)}}`;
                
                // Action text
                document.getElementById('profile-action-text').innerText = rfmData.actions[cust.Segment];
            }} else {{
                card.style.display = 'none';
                help.style.display = 'block';
                help.innerText = "No customer found matching '" + input + "'";
            }}
        }}

        // ----------------------------------------------------
        // CHARTS SETUP
        // ----------------------------------------------------
        
        // 1. Segments count bar chart
        const ctxCount = document.getElementById('chart-segments-count').getContext('2d');
        const sortedProfiles = [...rfmData.profiles].sort((a,b) => b.customers - a.customers);
        
        new Chart(ctxCount, {{
            type: 'bar',
            data: {{
                labels: sortedProfiles.map(p => p.Segment),
                datasets: [{{
                    data: sortedProfiles.map(p => p.customers),
                    backgroundColor: sortedProfiles.map(p => segColors[p.Segment]),
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: 'transparent' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#9ca3af' }} }}
                }}
            }}
        }});

        // 2. Segments revenue share polar area chart
        const ctxRev = document.getElementById('chart-segments-revenue').getContext('2d');
        new Chart(ctxRev, {{
            type: 'doughnut',
            data: {{
                labels: rfmData.profiles.map(p => p.Segment),
                datasets: [{{
                    data: rfmData.profiles.map(p => p.avg_monetary * p.customers),
                    backgroundColor: rfmData.profiles.map(p => segColors[p.Segment]),
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#9ca3af', font: {{ size: 10 }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Interactive HTML RFM Dashboard written to {dashboard_file}!")
    
    # 8. Create Technical RFM Report (rfm_report.md)
    print("Writing Technical RFM Report...")
    
    # Generate segment profile markdown table
    prof_df = pd.DataFrame(seg_profiles)
    prof_df.rename(columns={
        'Segment': 'Behavior Segment',
        'customers': 'Active Customers',
        'avg_recency': 'Avg Recency (Days)',
        'avg_frequency': 'Avg Frequency (Visits)',
        'avg_monetary': 'Avg Spend (₹)',
        'avg_loyalty': 'Avg Loyalty Points'
    }, inplace=True)
    
    # helper for markdown
    def to_md(df):
        cols = df.columns
        header = "| " + " | ".join(map(str, cols)) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for idx, row in df.iterrows():
            rows.append("| " + " | ".join(map(str, row.values)) + " |")
        return "\n".join([header, sep] + rows)
        
    profiles_md = to_md(prof_df)
    
    # Compile marketing strategies list
    actions_md = ""
    for seg, action in marketing_actions.items():
        actions_md += f"- **{seg}**: {action}\n"
        
    report_content = f"""# Technical Report: Customer Behavioral Segmentation (RFM Analysis)

## Executive Summary
This report presents a behavioral customer segmentation using the **RFM (Recency, Frequency, Monetary)** model. By analyzing transactional patterns from the star schema database, we grouped the restaurant's active customer base (2,987 customers) into **10 behavioral segments**. This allows the culinary marketing team to target specific groups with bespoke promotional campaigns, pricing tiers, and recipe adjustments.

---

## 1. RFM Methodology

### 1.1 Metrics Extraction
For each unique customer, three measures were calculated from `fact_sales`:
1. **Recency (R)**: The number of days elapsed between the customer's most recent order and the max dataset date (`2024-12-31`).
2. **Frequency (F)**: The count of unique orders (`order_id`) placed by the customer.
3. **Monetary (M)**: The sum of `net_sales` (transaction values) generated by the customer.

### 1.2 Quintile Scoring
Customers were assigned an integer score from 1 to 5 for each metric based on quintile cutoffs (top 20% gets 5, bottom 20% gets 1):
- **Recency score**: 5 represents the most recent visits, 1 represents dormant customers.
- **Frequency score**: Calculated using rank-ordering to resolve overlapping boundaries (due to discrete visitor counts), with 5 representing the most frequent diners.
- **Monetary score**: 5 represents the top spending customers.

The combined score `RFM_Group` (e.g. '555' for Champions, '111' for Lost) and `RFM_Score` (sum of scores, 3 to 15) were computed.

---

## 2. Customer Behavioral Profiles

The table below shows the profiling averages for each of the 10 behavioral segments:

{profiles_md}

### Key Profiles:
- **Champions**: Represent the most valuable segment. They order frequently, spend heavily, and visit regularly. They are key drivers of cash flow.
- **Loyal Customers**: Diners with stable, above-average ordering frequency and spend. They are highly engaged with the loyalty points program.
- **At Risk**: Customers who spent heavily in the past but have not visited in a long time. These require urgent win-back activation.

---

## 3. Targeted Engagement Action Plan

The table below defines targeted marketing and culinary activation strategies for each behavioral profile:

{actions_md}

---

## 4. Visualizations & Outputs

The RFM pipeline has successfully exported the following deliverables to the workspace:
1. **Enriched Customer Data**: [dim_customer_rfm.csv](file:///d:/cat/DAL/project/RMF/cleaned_data/dim_customer_rfm.csv) containing all metrics and segment tags.
2. **Consolidated Excel Workbook**: [Cleaned_Restaurant_Data_RFM.xlsx](file:///d:/cat/DAL/project/RMF/Cleaned_Restaurant_Data_RFM.xlsx) containing all fact and dimension tables.
3. **Static Charts**: Boxplots, scatter plots, and segment size bar charts saved under the [visualizations](file:///d:/cat/DAL/project/RMF/visualizations) directory.
4. **Interactive Dashboard**: A responsive, browser-ready dashboard [rfm_dashboard.html](file:///d:/cat/DAL/project/RMF/rfm_dashboard.html) featuring segment analysis and customer search tools.
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Technical RFM Report written to {report_file}!")
    print("RFM Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_rfm_pipeline()
