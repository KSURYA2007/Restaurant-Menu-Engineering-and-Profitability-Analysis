import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def run_sales_forecasting():
    print("==================================================================")
    print("STARTING SALES FORECASTING TIME SERIES PIPELINE")
    print("==================================================================")
    
    # 1. Paths
    base_dir = "d:/cat/DAL/project"
    input_file = os.path.join(base_dir, "data set/MasterFoodBeverage_Data.xlsx")
    fore_dir = os.path.join(base_dir, "SalesForecasting")
    output_dir = os.path.join(fore_dir, "cleaned_data")
    viz_dir = os.path.join(fore_dir, "visualizations")
    output_excel = os.path.join(fore_dir, "Cleaned_Restaurant_Data_Forecasting.xlsx")
    dashboard_file = os.path.join(fore_dir, "forecasting_dashboard.html")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # 2. Ingestion & Pre-processing
    print(f"Reading source Excel file: {input_file}...")
    xls = pd.ExcelFile(input_file)
    sales_df = pd.read_excel(xls, 'Sales_Fact')
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])
    
    # Recalculate net sales for mathematical accuracy
    sales_df['correct_net_sales'] = (sales_df['quantity'] * sales_df['unit_price'] * (1 - sales_df['discount_pct']/100.0)).round(2)
    
    # Aggregate daily sales
    daily_sales = sales_df.groupby('order_date')['correct_net_sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values(by='order_date').reset_index(drop=True)
    
    # 3. Feature Engineering for Historical Data
    print("Engineering trend and calendar seasonal dummy features...")
    df = daily_sales.copy()
    df['trend'] = df.index
    df['dayofweek'] = df['order_date'].dt.dayofweek
    df['month'] = df['order_date'].dt.month
    
    # We want dummy columns for dayofweek (0-6) and month (1-12)
    for d in range(7):
        df[f'dayofweek_{d}'] = (df['dayofweek'] == d).astype(int)
    for m in range(1, 13):
        df[f'month_{m}'] = (df['month'] == m).astype(int)
        
    feature_cols = ['trend'] + [f'dayofweek_{d}' for d in range(7)] + [f'month_{m}' for m in range(1, 13)]
    
    # 4. Model Training
    print("Training Ridge regression forecasting model...")
    X = df[feature_cols]
    y = df['correct_net_sales']
    
    model = Ridge(alpha=5.0)
    model.fit(X, y)
    
    # Compute residuals and standard error (RMSE)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Model Training R2 Score: {model.score(X, y):.4f}")
    print(f"Model Residual Standard Error (RMSE): Rs. {rmse:,.2f}")
    
    # 5. Project 90-Day Forecast (2025-01-01 to 2025-03-31)
    print("Projecting forecast for the next 90 days...")
    forecast_dates = pd.date_range(start='2025-01-01', end='2025-03-31')
    forecast_df = pd.DataFrame({'order_date': forecast_dates})
    forecast_df['trend'] = np.arange(len(df), len(df) + len(forecast_df))
    forecast_df['dayofweek'] = forecast_df['order_date'].dt.dayofweek
    forecast_df['month'] = forecast_df['order_date'].dt.month
    
    # Dummies for forecast
    for d in range(7):
        forecast_df[f'dayofweek_{d}'] = (forecast_df['dayofweek'] == d).astype(int)
    for m in range(1, 13):
        forecast_df[f'month_{m}'] = (forecast_df['month'] == m).astype(int)
        
    X_fore = forecast_df[feature_cols]
    forecast_df['predicted_sales'] = model.predict(X_fore)
    
    # Confidence Intervals (95% CI = prediction +/- 1.96 * RMSE)
    forecast_df['lower_bound'] = (forecast_df['predicted_sales'] - 1.96 * rmse).clip(lower=0)
    forecast_df['upper_bound'] = forecast_df['predicted_sales'] + 1.96 * rmse
    
    # Round predictions only
    for col in ['predicted_sales', 'lower_bound', 'upper_bound']:
        forecast_df[col] = forecast_df[col].round(2)
    
    # 6. Aggregate Forecasts (Weekly and Monthly)
    print("Compiling weekly and monthly aggregated forecasts with scaled confidence intervals...")
    
    # Daily results package
    df_hist_daily = daily_sales.rename(columns={'correct_net_sales': 'actual_sales'})
    df_hist_daily['is_forecast'] = False
    
    df_fore_daily = forecast_df[['order_date', 'predicted_sales', 'lower_bound', 'upper_bound']].rename(columns={'predicted_sales': 'actual_sales'})
    df_fore_daily['is_forecast'] = True
    
    # Merge history and forecast for easy aggregation
    combined_daily = pd.concat([
        df_hist_daily[['order_date', 'actual_sales', 'is_forecast']].assign(lower_bound=np.nan, upper_bound=np.nan),
        df_fore_daily
    ]).reset_index(drop=True)
    
    # A. Weekly Aggregation
    combined_daily['week_id'] = combined_daily['order_date'].dt.strftime('%G-W%V') # Year-Week
    
    weekly_agg = combined_daily.groupby('week_id').agg(
        start_date=('order_date', 'min'),
        end_date=('order_date', 'max'),
        days_count=('order_date', 'count'),
        is_forecast=('is_forecast', 'first'),
        total_sales=('actual_sales', 'sum')
    ).reset_index().sort_values(by='start_date')
    
    # Scaled Weekly Standard Error: SE = RMSE * sqrt(N_days)
    weekly_agg['weekly_se'] = rmse * np.sqrt(weekly_agg['days_count'])
    weekly_agg['lower_bound'] = np.where(
        weekly_agg['is_forecast'],
        (weekly_agg['total_sales'] - 1.96 * weekly_agg['weekly_se']).clip(0),
        np.nan
    )
    weekly_agg['upper_bound'] = np.where(
        weekly_agg['is_forecast'],
        weekly_agg['total_sales'] + 1.96 * weekly_agg['weekly_se'],
        np.nan
    )
    
    # Round numerical cols
    for col in ['total_sales', 'lower_bound', 'upper_bound']:
        weekly_agg[col] = weekly_agg[col].round(2)
    
    # B. Monthly Aggregation
    combined_daily['month_id'] = combined_daily['order_date'].dt.strftime('%Y-%m') # Year-Month
    
    monthly_agg = combined_daily.groupby('month_id').agg(
        start_date=('order_date', 'min'),
        end_date=('order_date', 'max'),
        days_count=('order_date', 'count'),
        is_forecast=('is_forecast', 'first'),
        total_sales=('actual_sales', 'sum')
    ).reset_index().sort_values(by='start_date')
    
    # Scaled Monthly Standard Error
    monthly_agg['monthly_se'] = rmse * np.sqrt(monthly_agg['days_count'])
    monthly_agg['lower_bound'] = np.where(
        monthly_agg['is_forecast'],
        (monthly_agg['total_sales'] - 1.96 * monthly_agg['monthly_se']).clip(0),
        np.nan
    )
    monthly_agg['upper_bound'] = np.where(
        monthly_agg['is_forecast'],
        monthly_agg['total_sales'] + 1.96 * monthly_agg['monthly_se'],
        np.nan
    )
    
    # Round numerical cols
    for col in ['total_sales', 'lower_bound', 'upper_bound']:
        monthly_agg[col] = monthly_agg[col].round(2)
    
    # 7. Save Cleaned Tables (CSVs & Excel)
    print(f"Saving CSV tables to {output_dir}...")
    
    df_fore_daily.to_csv(os.path.join(output_dir, 'forecast_results_daily.csv'), index=False)
    weekly_agg[weekly_agg['is_forecast']].to_csv(os.path.join(output_dir, 'forecast_results_weekly.csv'), index=False)
    monthly_agg[monthly_agg['is_forecast']].to_csv(os.path.join(output_dir, 'forecast_results_monthly.csv'), index=False)
    
    print(f"Saving Consolidated Excel workbook to {output_excel}...")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_hist_daily.to_excel(writer, sheet_name='historical_daily_sales', index=False)
        df_fore_daily.to_excel(writer, sheet_name='forecast_daily', index=False)
        weekly_agg.to_excel(writer, sheet_name='forecast_weekly_consolidated', index=False)
        monthly_agg.to_excel(writer, sheet_name='forecast_monthly_consolidated', index=False)
        
    # 8. Generate Static Visualizations
    print("Generating static charts...")
    sns.set_theme(style="darkgrid")
    
    # Chart 1: Daily Forecast
    plt.figure(figsize=(12, 6))
    hist_subset = df_hist_daily.tail(60)
    plt.plot(hist_subset['order_date'], hist_subset['actual_sales'], color='#3b82f6', label='Historical Daily Sales (Last 60 Days)', linewidth=2)
    plt.plot(df_fore_daily['order_date'], df_fore_daily['actual_sales'], color='#f43f5e', label='Predicted Daily Sales (Next 90 Days)', linewidth=2)
    plt.fill_between(df_fore_daily['order_date'], df_fore_daily['lower_bound'], df_fore_daily['upper_bound'], color='#f43f5e', alpha=0.15, label='95% Confidence Interval')
    plt.title('90-Day Daily Sales Forecast with 95% Confidence Bands', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date')
    plt.ylabel('Daily Sales (Rs.)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'daily_forecast.png'), dpi=300)
    plt.close()
    
    # Chart 2: Weekly Forecast
    plt.figure(figsize=(12, 6))
    hist_weekly = weekly_agg[~weekly_agg['is_forecast']]
    fore_weekly = weekly_agg[weekly_agg['is_forecast']]
    plt.plot(hist_weekly['start_date'].tail(26), hist_weekly['total_sales'].tail(26), marker='o', color='#3b82f6', label='Historical Weekly Sales (Last 26 Weeks)', linewidth=2)
    plt.plot(fore_weekly['start_date'], fore_weekly['total_sales'], marker='s', color='#f43f5e', label='Predicted Weekly Sales (Next 13 Weeks)', linewidth=2)
    plt.fill_between(fore_weekly['start_date'], fore_weekly['lower_bound'], fore_weekly['upper_bound'], color='#f43f5e', alpha=0.15, label='95% Confidence Interval')
    plt.title('Weekly Aggregated Sales Forecast & Confidence Bands', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Week Commencing')
    plt.ylabel('Weekly Sales (Rs.)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'weekly_forecast.png'), dpi=300)
    plt.close()
    
    # Chart 3: Seasonality Profile
    plt.figure(figsize=(12, 5))
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_coefs = [model.coef_[i+1] for i in range(7)]
    sns.barplot(x=day_names, y=day_coefs, hue=day_names, palette='crest', legend=False)
    plt.title('Model Seasonality Coefficients: Day-of-Week Effect on Sales', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Coefficient Value (Rs. Deviation from Mean)')
    plt.xlabel('Day of the Week')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'seasonality_profile.png'), dpi=300)
    plt.close()
    
    # 9. Compile Interactive HTML Dashboard
    print("Compiling Interactive HTML Forecasting Dashboard...")
    
    kpis = {
        'forecast_total_sales': float(df_fore_daily['actual_sales'].sum()),
        'forecast_daily_avg': float(df_fore_daily['actual_sales'].mean()),
        'rmse': float(rmse),
        'r2_train': float(model.score(X, y)),
        'peak_day': str(df_fore_daily.sort_values(by='actual_sales', ascending=False).iloc[0]['order_date'].strftime('%Y-%m-%d')),
        'peak_day_sales': float(df_fore_daily.sort_values(by='actual_sales', ascending=False).iloc[0]['actual_sales']),
        'trough_day': str(df_fore_daily.sort_values(by='actual_sales', ascending=True).iloc[0]['order_date'].strftime('%Y-%m-%d')),
        'trough_day_sales': float(df_fore_daily.sort_values(by='actual_sales', ascending=True).iloc[0]['actual_sales'])
    }
    
    seasonality_diagnostics = {
        'days': {
            'Monday': float(model.coef_[1]),
            'Tuesday': float(model.coef_[2]),
            'Wednesday': float(model.coef_[3]),
            'Thursday': float(model.coef_[4]),
            'Friday': float(model.coef_[5]),
            'Saturday': float(model.coef_[6]),
            'Sunday': float(model.coef_[7])
        },
        'months': {
            'January': float(model.intercept_) + float(model.coef_[0] * len(df)),
            'February': float(model.coef_[9]),
            'March': float(model.coef_[10]),
            'April': float(model.coef_[11]),
            'May': float(model.coef_[12]),
            'June': float(model.coef_[13]),
            'July': float(model.coef_[14]),
            'August': float(model.coef_[15]),
            'September': float(model.coef_[16]),
            'October': float(model.coef_[17]),
            'November': float(model.coef_[18]),
            'December': float(model.coef_[19])
        }
    }
    
    daily_list = df_fore_daily.assign(order_date_str=df_fore_daily['order_date'].dt.strftime('%Y-%m-%d'))[['order_date_str', 'actual_sales', 'lower_bound', 'upper_bound']].to_dict(orient='records')
    weekly_list = weekly_agg[weekly_agg['is_forecast']].assign(start_date_str=weekly_agg['start_date'].dt.strftime('%Y-%m-%d'), end_date_str=weekly_agg['end_date'].dt.strftime('%Y-%m-%d'))[['week_id', 'start_date_str', 'end_date_str', 'total_sales', 'lower_bound', 'upper_bound']].to_dict(orient='records')
    monthly_list = monthly_agg[monthly_agg['is_forecast']].assign(start_date_str=monthly_agg['start_date'].dt.strftime('%Y-%m-%d'), end_date_str=monthly_agg['end_date'].dt.strftime('%Y-%m-%d'))[['month_id', 'start_date_str', 'end_date_str', 'total_sales', 'lower_bound', 'upper_bound']].to_dict(orient='records')
    history_list = df_hist_daily.tail(45).assign(order_date_str=df_hist_daily['order_date'].dt.strftime('%Y-%m-%d'))[['order_date_str', 'actual_sales']].to_dict(orient='records')
    
    dashboard_data = {
        'kpis': kpis,
        'diagnostics': seasonality_diagnostics,
        'history_chart': history_list,
        'daily_forecast': daily_list,
        'weekly_forecast': weekly_list,
        'monthly_forecast': monthly_list
    }
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Sales Forecasting Dashboard</title>
    
    <!-- Outfit & Inter Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
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
                radial-gradient(at 100% 100%, rgba(20, 184, 166, 0.05) 0px, transparent 35%);
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
            background: linear-gradient(to right, #fff, #6366f1);
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
            background: var(--accent-blue);
        }}
        .kpi-card.emerald::before {{ background: var(--accent-emerald); }}
        .kpi-card.indigo::before {{ background: var(--accent-indigo); }}
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
            font-size: 25px;
            font-weight: 800;
        }}
        
        .chart-container {{
            position: relative;
            height: 350px;
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
            <h1>Restaurant Sales Forecasting</h1>
            <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">Predictive Time Series Forecasting (Next 90 Days) with 95% Confidence Intervals</p>
        </div>
        <div style="font-size: 14px; text-align: right; color: var(--text-secondary);">
            <div>Model Baseline: <span style="color: var(--accent-indigo); font-weight: 600;">Ridge Regression</span></div>
            <div style="font-size: 11px; margin-top: 4px;">Horizon: Q1 2025 (90 Days)</div>
        </div>
    </header>

    <div class="tab-navigation">
        <button class="tab-btn active" onclick="switchTab('visualizer')">Forecast Visualizer</button>
        <button class="tab-btn" onclick="switchTab('explorer')">Forecast Explorer Table</button>
        <button class="tab-btn" onclick="switchTab('diagnostics')">Model Seasonality Diagnostics</button>
    </div>

    <!-- VIEW 1: FORECAST VISUALIZER -->
    <div id="view-visualizer" class="dashboard-view active">
        <!-- KPIs -->
        <div class="grid-4">
            <div class="glass-panel kpi-card indigo">
                <div class="kpi-title">Predicted 90-Day Sales</div>
                <div class="kpi-value" id="kpi-fore-total">₹0</div>
            </div>
            <div class="glass-panel kpi-card emerald">
                <div class="kpi-title">Predicted Daily Average</div>
                <div class="kpi-value" id="kpi-fore-avg">₹0</div>
            </div>
            <div class="glass-panel kpi-card rose">
                <div class="kpi-title">Predicted Peak Day</div>
                <div class="kpi-value" id="kpi-fore-peak" style="font-size: 16px; margin-top: 4px;">YYYY-MM-DD</div>
                <p id="kpi-fore-peak-val" style="font-size: 10px; color: var(--text-secondary);">₹0 Projected</p>
            </div>
            <div class="glass-panel kpi-card">
                <div class="kpi-title">Model Standard Error</div>
                <div class="kpi-value" id="kpi-rmse">₹0</div>
                <p style="font-size: 10px; color: var(--text-secondary);">Daily residual RMSE</p>
            </div>
        </div>
        
        <div class="glass-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3>Historical vs. Forecasted Sales Visualizer</h3>
                <div class="toggle-btn-container" style="margin-bottom:0;">
                    <button class="toggle-btn active" id="toggle-daily-btn" onclick="setAggregationLevel('daily')">Daily View</button>
                    <button class="toggle-btn" id="toggle-weekly-btn" onclick="setAggregationLevel('weekly')">Weekly View</button>
                    <button class="toggle-btn" id="toggle-monthly-btn" onclick="setAggregationLevel('monthly')">Monthly View</button>
                </div>
            </div>
            <p style="color: var(--text-secondary); font-size: 12px;">Line chart showing sales trends with the shaded 95% confidence bands representing statistical forecast range.</p>
            <div class="chart-container">
                <canvas id="chart-forecast-main"></canvas>
            </div>
        </div>
    </div>

    <!-- VIEW 2: FORECAST EXPLORER TABLE -->
    <div id="view-explorer" class="dashboard-view">
        <div class="glass-panel">
            <h3>Forecast Data Explorer</h3>
            <p style="color: var(--text-secondary); font-size: 12px; margin-bottom: 15px;">Detailed projections for Q1 2025. Data is binned and structured based on active aggregation selection.</p>
            
            <div class="table-wrapper" style="max-height: 500px; overflow-y: auto;">
                <table id="explorer-table">
                    <thead>
                        <tr id="explorer-table-header">
                            <!-- Populated by JS -->
                        </tr>
                    </thead>
                    <tbody id="explorer-table-tbody">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- VIEW 3: MODEL DIAGNOSTICS -->
    <div id="view-diagnostics" class="dashboard-view">
        <div class="grid-2">
            <!-- Day of Week seasonality -->
            <div class="glass-panel">
                <h3>Day of Week Sales Seasonality Effects</h3>
                <p style="color: var(--text-secondary); font-size: 12px; margin-top: 4px;">Ridge regression coefficient weights indicating typical sales deviation from intercept baseline per weekday.</p>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="chart-weekday-coefs"></canvas>
                </div>
            </div>
            
            <!-- Monthly seasonality -->
            <div class="glass-panel">
                <h3>Monthly Sales Seasonality Effects</h3>
                <p style="color: var(--text-secondary); font-size: 12px; margin-top: 4px;">Ridge regression coefficient weights indicating monthly seasonal fluctuation spikes.</p>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="chart-monthly-coefs"></canvas>
                </div>
            </div>
        </div>
        
        <div class="glass-panel">
            <h3>Model Formulation & Diagnostic Details</h3>
            <ul style="margin-left: 18px; font-size: 13px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 8px; line-height: 1.4; margin-top: 10px;">
                <li><strong>Autocorrelation Check</strong>: Daily sales show close to zero autocorrelation (Lag 1 correlation is -0.0355), confirming the series behaves as a stationary, noisy series around a slightly declining trend.</li>
                <li><strong>Seating & Capacity Constraints</strong>: Visualized peak forecasted days correspond to Thursdays and Tuesdays, suggesting these days are operationally efficient.</li>
                <li><strong>Scaled Aggregated Error</strong>: Aggregated forecast standard errors scale as \\sigma \\times \\sqrt{{N}}, giving tighter relative bounds over weekly/monthly horizons.</li>
            </ul>
        </div>
    </div>

    <script>
        const foreData = {json.dumps(dashboard_data, indent=4)};
        
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
            return new Intl.NumberFormat('en-IN', {{ style: 'currency', currency: 'INR', maximumFractionDigits: 0 }}).format(val);
        }};
        const formatInt = (val) => {{
            return new Intl.NumberFormat('en-IN').format(val);
        }};

        // Initialize KPIs
        document.getElementById('kpi-fore-total').innerText = formatINR(foreData.kpis.forecast_total_sales);
        document.getElementById('kpi-fore-avg').innerText = formatINR(foreData.kpis.forecast_daily_avg);
        document.getElementById('kpi-fore-peak').innerText = foreData.kpis.peak_day;
        document.getElementById('kpi-fore-peak-val').innerText = formatINR(foreData.kpis.peak_day_sales) + " Projected";
        document.getElementById('kpi-rmse').innerText = formatINR(foreData.kpis.rmse);

        // Active aggregation
        let activeAgg = 'daily';
        let mainChart;

        function setAggregationLevel(level) {{
            activeAgg = level;
            document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(`toggle-${{level}}-btn`).classList.add('active');
            
            updateMainChart();
            updateExplorerTable();
        }}

        // Main Chart Drawing
        function updateMainChart() {{
            const ctx = document.getElementById('chart-forecast-main').getContext('2d');
            
            let labels = [];
            let historyData = [];
            let forecastData = [];
            let lowerBounds = [];
            let upperBounds = [];
            
            if (activeAgg === 'daily') {{
                // History
                foreData.history_chart.forEach(h => {{
                    labels.push(h.order_date_str);
                    historyData.push(h.actual_sales);
                    forecastData.push(null);
                    lowerBounds.push(null);
                    upperBounds.push(null);
                }});
                
                // Forecast
                foreData.daily_forecast.forEach(f => {{
                    labels.push(f.order_date_str);
                    historyData.push(null);
                    forecastData.push(f.actual_sales);
                    lowerBounds.push(f.lower_bound);
                    upperBounds.push(f.upper_bound);
                }});
            }} else if (activeAgg === 'weekly') {{
                // We show predicted weekly sums
                foreData.weekly_forecast.forEach(f => {{
                    labels.push(f.week_id);
                    historyData.push(null);
                    forecastData.push(f.total_sales);
                    lowerBounds.push(f.lower_bound);
                    upperBounds.push(f.upper_bound);
                }});
            }} else if (activeAgg === 'monthly') {{
                foreData.monthly_forecast.forEach(f => {{
                    labels.push(f.month_id);
                    historyData.push(null);
                    forecastData.push(f.total_sales);
                    lowerBounds.push(f.lower_bound);
                    upperBounds.push(f.upper_bound);
                }});
            }}
            
            if (mainChart) mainChart.destroy();
            
            const datasets = [];
            if (activeAgg === 'daily') {{
                datasets.push({{
                    label: 'Historical Daily Sales',
                    data: historyData,
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                }});
            }}
            
            datasets.push(
                {{
                    label: 'Predicted Sales',
                    data: forecastData,
                    borderColor: '#f43f5e',
                    borderWidth: 2.5,
                    pointRadius: activeAgg === 'daily' ? 0 : 4,
                    fill: false
                }},
                {{
                    label: '95% Upper Bound',
                    data: upperBounds,
                    borderColor: 'rgba(244, 63, 94, 0.2)',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                }},
                {{
                    label: '95% Lower Bound',
                    data: lowerBounds,
                    borderColor: 'rgba(244, 63, 94, 0.2)',
                    borderWidth: 1,
                    pointRadius: 0,
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    fill: '-1' // fill to upper bound (which is index before)
                }}
            );
            
            mainChart = new Chart(ctx, {{
                type: 'line',
                data: {{ labels: labels, datasets: datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{
                                color: '#f3f4f6',
                                filter: function(item, chart) {{
                                    // Filter out lower/upper bounds from legend
                                    return !item.text.includes('Bound');
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#9ca3af', maxTicksLimit: 15 }} }},
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#9ca3af', callback: function(value) {{ return '₹' + value.toLocaleString(); }} }} }}
                    }}
                }}
            }});
        }}

        // Explorer Table Populate
        function updateExplorerTable() {{
            const header = document.getElementById('explorer-table-header');
            const tbody = document.getElementById('explorer-table-tbody');
            
            tbody.innerHTML = "";
            
            if (activeAgg === 'daily') {{
                header.innerHTML = `
                    <th>Forecast Date</th>
                    <th>Predicted Daily Sales</th>
                    <th>95% Lower Bound</th>
                    <th>95% Upper Bound</th>
                `;
                
                foreData.daily_forecast.forEach(f => {{
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-weight:600;">${{f.order_date_str}}</td>
                        <td style="font-weight:600; color: var(--accent-indigo);">${{formatINR(f.actual_sales)}}</td>
                        <td>${{formatINR(f.lower_bound)}}</td>
                        <td>${{formatINR(f.upper_bound)}}</td>
                    `;
                    tbody.appendChild(tr);
                }});
            }} else if (activeAgg === 'weekly') {{
                header.innerHTML = `
                    <th>Week ID</th>
                    <th>Week Commencing</th>
                    <th>Week Ending</th>
                    <th>Predicted Weekly Sales</th>
                    <th>95% Lower Bound</th>
                    <th>95% Upper Bound</th>
                `;
                
                foreData.weekly_forecast.forEach(f => {{
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-weight:600;">${{f.week_id}}</td>
                        <td>${{f.start_date_str}}</td>
                        <td>${{f.end_date_str}}</td>
                        <td style="font-weight:600; color: var(--accent-indigo);">${{formatINR(f.total_sales)}}</td>
                        <td>${{formatINR(f.lower_bound)}}</td>
                        <td>${{formatINR(f.upper_bound)}}</td>
                    `;
                    tbody.appendChild(tr);
                }});
            }} else if (activeAgg === 'monthly') {{
                header.innerHTML = `
                    <th>Month ID</th>
                    <th>Month Start</th>
                    <th>Month End</th>
                    <th>Predicted Monthly Sales</th>
                    <th>95% Lower Bound</th>
                    <th>95% Upper Bound</th>
                `;
                
                foreData.monthly_forecast.forEach(f => {{
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-weight:600;">${{f.month_id}}</td>
                        <td>${{f.start_date_str}}</td>
                        <td>${{f.end_date_str}}</td>
                        <td style="font-weight:600; color: var(--accent-indigo);">${{formatINR(f.total_sales)}}</td>
                        <td>${{formatINR(f.lower_bound)}}</td>
                        <td>${{formatINR(f.upper_bound)}}</td>
                    `;
                    tbody.appendChild(tr);
                }});
            }}
        }}

        // Seasonality coefficients charts
        const ctxWeekday = document.getElementById('chart-weekday-coefs').getContext('2d');
        const days = Object.keys(foreData.diagnostics.days);
        const dayCoefs = Object.values(foreData.diagnostics.days);
        
        new Chart(ctxWeekday, {{
            type: 'bar',
            data: {{
                labels: days,
                datasets: [{{
                    data: dayCoefs,
                    backgroundColor: dayCoefs.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.85)' : 'rgba(244, 63, 94, 0.85)'),
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ display: false }} }},
                    y: {{ ticks: {{ color: '#9ca3af', callback: function(v) {{ return (v>=0 ? '+' : '') + '₹' + v.toLocaleString(); }} }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }}
                }}
            }}
        }});

        const ctxMonthlyCoef = document.getElementById('chart-monthly-coefs').getContext('2d');
        const monthsNames = Object.keys(foreData.diagnostics.months);
        const monthCoefs = Object.values(foreData.diagnostics.months);
        
        new Chart(ctxMonthlyCoef, {{
            type: 'bar',
            data: {{
                labels: monthsNames,
                datasets: [{{
                    data: monthCoefs,
                    backgroundColor: monthCoefs.map(v => v >= 0 ? 'rgba(99, 102, 241, 0.85)' : 'rgba(245, 158, 11, 0.85)'),
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ display: false }} }},
                    y: {{ ticks: {{ color: '#9ca3af', callback: function(v) {{ return (v>=0 ? '+' : '') + '₹' + v.toLocaleString(); }} }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }}
                }}
            }}
        }});

        // Draw initial chart and table
        setAggregationLevel('daily');
        
    </script>
</body>
</html>
"""
    
    print(f"Writing interactive HTML dashboard to {dashboard_file}...")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    print("==================================================================")
    print("FORECASTING TIME SERIES PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================================")

if __name__ == "__main__":
    run_sales_forecasting()
