import pandas as pd
import numpy as np

file_path = r'd:/cat/DAL/project/data set/MasterFoodBeverage_Data.xlsx'
print(f"Loading data from {file_path}...")
df = pd.read_excel(file_path)

total_original_rows = len(df)
print(f"Total original rows: {total_original_rows}")

# Track indices of bad rows for each category
bad_indices = {
    'missing_values': set(),
    'duplicates': set(),
    'out_of_range': set(),
    'logical_errors': set(),
    'inconsistent_formats': set()
}

# 1. Missing Values
missing_mask = df.isnull().any(axis=1)
bad_indices['missing_values'].update(df[missing_mask].index)

# 2. Duplicates
duplicate_mask = df.duplicated(keep=False)
bad_indices['duplicates'].update(df[duplicate_mask].index)

# 3. Out of Range / Invalid values
out_of_range_mask = pd.Series(False, index=df.index)
if 'quantity' in df.columns:
    out_of_range_mask |= (df['quantity'] <= 0)
if 'unit_price' in df.columns:
    out_of_range_mask |= (df['unit_price'] < 0)
if 'discount_pct' in df.columns:
    out_of_range_mask |= (df['discount_pct'] < 0) | (df['discount_pct'] > 1)
if 'customer_rating' in df.columns:
    out_of_range_mask |= (df['customer_rating'] < 1) | (df['customer_rating'] > 5)
bad_indices['out_of_range'].update(df[out_of_range_mask].index)

# 4. Logical Errors
logical_error_mask = pd.Series(False, index=df.index)
# Example logical checks (using np.isclose to handle floating point precision)
if all(c in df.columns for c in ['quantity', 'unit_price', 'gross_sales']):
    logical_error_mask |= ~np.isclose(df['quantity'] * df['unit_price'], df['gross_sales'], rtol=1e-2, atol=1e-2) & df['gross_sales'].notnull()

if all(c in df.columns for c in ['gross_sales', 'discount_amt', 'net_sales']):
    logical_error_mask |= ~np.isclose(df['gross_sales'] - df['discount_amt'], df['net_sales'], rtol=1e-2, atol=1e-2) & df['net_sales'].notnull()

if all(c in df.columns for c in ['net_sales', 'cogs', 'profit']):
    logical_error_mask |= ~np.isclose(df['net_sales'] - df['cogs'], df['profit'], rtol=1e-2, atol=1e-2) & df['profit'].notnull()
bad_indices['logical_errors'].update(df[logical_error_mask].index)

# 5. Inconsistent formats
inconsistent_mask = pd.Series(False, index=df.index)
# E.g. order_type has mixed casing or leading/trailing spaces
if 'order_type' in df.columns:
    # Check if order type isn't a string, or has leading/trailing spaces, or isn't uppercase/lowercase consistently
    # Simplest check: strings that change when stripped
    is_string = df['order_type'].apply(lambda x: isinstance(x, str))
    needs_strip = is_string & (df['order_type'] != df['order_type'].str.strip())
    inconsistent_mask |= needs_strip

if 'order_date' in df.columns:
    # check if dates are wildly out of range (e.g. before 1900 or after 2100)
    if pd.api.types.is_datetime64_any_dtype(df['order_date']):
        inconsistent_mask |= (df['order_date'].dt.year < 1900) | (df['order_date'].dt.year > 2100)

bad_indices['inconsistent_formats'].update(df[inconsistent_mask].index)

print("\n--- BAD DATA REPORT ---")
all_bad_indices = set()
for category, indices in bad_indices.items():
    print(f"- {category.replace('_', ' ').title()}: {len(indices)} rows")
    all_bad_indices.update(indices)

total_bad_rows = len(all_bad_indices)
print(f"\nTotal distinct bad rows (a row may have multiple issues): {total_bad_rows}")
print(f"Total rows with NO issues (Good Data): {total_original_rows - total_bad_rows}")

# Simulate cleaning to get converted good data
# Strategy: We will drop all rows that we identified as bad
clean_df = df.drop(index=list(all_bad_indices))
final_good_rows = len(clean_df)

print("\n--- CLEANING SUMMARY ---")
print(f"Rows dropped during strict cleaning (all bad rows removed): {total_original_rows - final_good_rows}")
print(f"Total rows of GOOD DATA after cleaning process: {final_good_rows}")
