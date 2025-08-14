import pandas as pd
import numpy as np
from pathlib import Path

PROVINCES = {
    'Alberta': 'Alberta',
    'British Columbia': 'British Columbia',
    'Manitoba': 'Manitoba', 
    'New Brunswick': 'New Brunswick',
    'Newfoundland and Labrador': 'Newfoundland and Labrador',
    'Nova Scotia': 'Nova Scotia',
    'Ontario': 'Ontario',
    'Prince Edward Island': 'Prince Edward Island',
    'Quebec': 'Quebec',
    'Saskatchewan': 'Saskatchewan'
}

TARGET_PROVINCES = list(PROVINCES.keys())

# Find where each province's data starts in the CSV
def find_province_sections(df):
    province_sections = {}
    for i, row in df.iterrows():
        # Check first column for province names
        first_col = str(row.iloc[0]).strip()
        for province in TARGET_PROVINCES:
            if province.lower() in first_col.lower():
                province_sections[province] = i
                break
    return province_sections

# Find where the actual year data starts for a province
def find_data_start_for_province(df, province_start_row):
    # Look for the header row (Year, Hospitals, etc.)
    for i in range(province_start_row, min(province_start_row + 15, len(df))):
        row_str = ' '.join([str(cell) for cell in df.iloc[i] if pd.notna(cell)])
        if 'Year' in row_str and 'Hospitals' in row_str:
            return i + 1
    return None

# Extract spending data for one province
def extract_province_data(df, province_name, start_row):
    data_rows = []
    for i in range(start_row, len(df)):
        row = df.iloc[i]
        first_col = str(row.iloc[0]).strip()

        # Check if we've hit another province section
        if any(prov.lower() in first_col.lower() for prov in TARGET_PROVINCES if prov != province_name):
            break

        # Check if this is a year row
        if first_col.replace(' f', '').replace('f', '').isdigit():
            year_str = first_col.replace(' f', '').replace('f', '')
            year = int(year_str)
            if 2008 <= year <= 2024:
                # Extract spending values
                row_data = {
                    'province': province_name,
                    'year': year,
                    'hospitals': clean_spending_value(row.iloc[1]),
                    'other_institutions': clean_spending_value(row.iloc[2]),
                    'physicians': clean_spending_value(row.iloc[3]),
                    'other_professionals': clean_spending_value(row.iloc[4]),
                    'drugs': clean_spending_value(row.iloc[5]),
                    'public_health': clean_spending_value(row.iloc[6]),
                    'administration': clean_spending_value(row.iloc[7]),
                    'other_hcc': clean_spending_value(row.iloc[8]),
                    'other_net': clean_spending_value(row.iloc[9]),
                    'covid19': clean_spending_value(row.iloc[10]),
                    'subtotal': clean_spending_value(row.iloc[11]),
                    'capital': clean_spending_value(row.iloc[12]),
                    'total_spending_per_capita': clean_spending_value(row.iloc[13])
                }
                data_rows.append(row_data)

        elif first_col == '' or pd.isna(row.iloc[0]):
            continue
        elif 'Source:' in first_col or 'Note:' in first_col:
            break
    
    result_df = pd.DataFrame(data_rows)
    return result_df

# Clean a spending value from the CSV
def clean_spending_value(value):
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    # Handle missing values
    if value_str in ['—', 'n/a', 'N/A', '', 'nan', 'NaN']:
        return np.nan
    
    try:
        clean_value = value_str.replace('"', '').replace(',', '')
        return float(clean_value)
    except (ValueError, TypeError):
        return np.nan

# Clean healthcare spending breakdown data from consolidated CSV
def clean_spending_breakdown_data(input_file: str, output_dir: str):
    print("Dataset2: Cleaning healthcare spending breakdown data...")
    df = pd.read_csv(input_file, header=None) 
    
    province_sections = find_province_sections(df)
    if not province_sections:
        print("Error: Could not find any provinces in the data")
        return pd.DataFrame()
    
    all_data = []

    for province, start_row in province_sections.items():
        data_start = find_data_start_for_province(df, start_row)
        if data_start is None:
            continue

        province_data = extract_province_data(df, province, data_start)
        if not province_data.empty:
            all_data.append(province_data)
    
    if not all_data:
        print("No data was successfully extracted")
        return pd.DataFrame()
    
    analysis_table = pd.concat(all_data, ignore_index=True)
    
    # Round spending values
    spending_columns = [col for col in analysis_table.columns if col not in ['province', 'year']]
    for col in spending_columns:
        analysis_table[col] = analysis_table[col].round(2)
    
    analysis_table = analysis_table.sort_values(['province', 'year']).reset_index(drop=True)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'spending_breakdown_for_analysis.csv'
    analysis_table.to_csv(output_file, index=False)
    
    print(f"Spending breakdown cleaned: {len(analysis_table)} records saved to {output_file}")
    
    return analysis_table

if __name__ == "__main__":
    INPUT_FILE = "../data/raw/dataset2_total_expenditure.csv"
    OUTPUT_DIR = "../data/processed"
    
    breakdown_df = clean_spending_breakdown_data(INPUT_FILE, OUTPUT_DIR)