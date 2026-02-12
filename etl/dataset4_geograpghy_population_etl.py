import pandas as pd
import numpy as np
from pathlib import Path

PROVINCE_MAPPING = {
    'British Columbia [Province]': 'British Columbia',
    'Alberta [Province]': 'Alberta', 
    'Saskatchewan [Province]': 'Saskatchewan',
    'Manitoba [Province]': 'Manitoba',
    'Ontario [Province]': 'Ontario',
    'Newfoundland and Labrador [Province]': 'Newfoundland and Labrador',
    'Prince Edward Island [Province]': 'Prince Edward Island',
    'Nova Scotia [Province]': 'Nova Scotia',
    'New Brunswick [Province]': 'New Brunswick',
    'Quebec [Province]': 'Quebec'
}

KEY_CHARACTERISTICS = {
    'Population, 2021': 'population_2021',
    'Population, 2016': 'population_2016', 
    'Population percentage change, 2016 to 2021': 'population_growth_pct',
    'Population density per square kilometre': 'population_density_per_km2',
    'Land area in square kilometres': 'land_area_km2',
    '65 years and over': 'population_65_plus',
    'Average age of the population': 'average_age',
    'Median age of the population': 'median_age'
}

# Find provinces in row 2 and their corresponding Total columns in row 4
def find_provinces_and_total_columns(df):
    province_row = df.iloc[1]
    header_row = df.iloc[3]
    
    provinces_found = {}
    
    for col_idx, province_name in enumerate(province_row):
        if pd.notna(province_name) and '[Province]' in str(province_name):
            province_str = str(province_name).strip()
            
            if province_str in PROVINCE_MAPPING:
                standard_name = PROVINCE_MAPPING[province_str]
                
                # Find the "Total" column for this province
                total_col = None
                for offset in range(0, 20):
                    check_col = col_idx + offset
                    if check_col < len(header_row):
                        header_val = str(header_row.iloc[check_col]).strip() if pd.notna(header_row.iloc[check_col]) else ""
                        if header_val == "Total":
                            total_col = check_col
                            break
                
                if total_col is not None:
                    provinces_found[standard_name] = total_col
    
    return provinces_found

# Extract data for a specific characteristic across all provinces
def extract_characteristic_data(df, characteristic_name, province_total_cols):
    characteristic_data = {}
    
    for row_idx in range(4, len(df)):
        char_cell = df.iloc[row_idx, 1]
        
        if pd.notna(char_cell):
            char_str = str(char_cell).strip().strip('"')
            
            if char_str == characteristic_name:
                for province, total_col in province_total_cols.items():
                    if total_col < len(df.columns):
                        value = df.iloc[row_idx, total_col]
                        clean_value = clean_numeric_value(value)
                        characteristic_data[province] = clean_value
                break
    
    return characteristic_data

# Clean and convert a value to numeric
def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # I will have to handle some common non-numeric indicators
    if value_str in ['...', 'x', 'F', '..F', 'E', '..E', '', 'nan']:
        return np.nan
    
    try:
        clean_value = value_str.replace(',', '')
        return float(clean_value)
    except (ValueError, TypeError):
        return np.nan

# Calculate additional variables from the extracted data
def calculate_derived_variables(province_data):
    # Calculate rural/urban estimates based on population density (Still thinking if I will use this for my analysis)
    if 'population_density_per_km2' in province_data and pd.notna(province_data['population_density_per_km2']):
        density = province_data['population_density_per_km2']
    
        if density < 1:
            urban_pct = 20  # Very rural
        elif density < 10:
            urban_pct = 40  # Rural  
        elif density < 100:
            urban_pct = 70  # Mixed
        else:
            urban_pct = 90  # Urban
            
        province_data['rural_percent_estimate'] = 100 - urban_pct
        province_data['urban_percent_estimate'] = urban_pct
    
    # Calculate 65+ percentage
    if ('population_65_plus' in province_data and 'population_2021' in province_data and
        pd.notna(province_data['population_65_plus']) and pd.notna(province_data['population_2021'])):
        
        pct_65_plus = (province_data['population_65_plus'] / province_data['population_2021']) * 100
        province_data['percent_65_plus'] = round(pct_65_plus, 1)
    
    return province_data

# Clean demographics and geographic data from multiple files
def clean_demographics_data(input_files: list, output_dir: str):
    print("Dataset4: Cleaning demographics and geography data...")
    
    all_provinces_data = []
    
    for file_idx, input_file in enumerate(input_files):
        df = pd.read_csv(input_file, header=None)
        
        province_total_cols = find_provinces_and_total_columns(df)
        
        if not province_total_cols:
            continue
        
        file_province_data = {}
        
        for province in province_total_cols.keys():
            file_province_data[province] = {
                'province': province,
                'year': 2021
            }
        
        # Extract each characteristic
        for char_name, var_name in KEY_CHARACTERISTICS.items():
            char_data = extract_characteristic_data(df, char_name, province_total_cols)
            
            for province, value in char_data.items():
                if province in file_province_data:
                    file_province_data[province][var_name] = value
        
        # Calculate derived variables
        for province in file_province_data.keys():
            file_province_data[province] = calculate_derived_variables(file_province_data[province])
        
        all_provinces_data.extend(list(file_province_data.values()))
    
    if not all_provinces_data:
        print("No data was successfully extracted from any files")
        return pd.DataFrame()
    
    analysis_table = pd.DataFrame(all_provinces_data)
    
    # Round numeric columns
    numeric_cols = [col for col in analysis_table.columns if col not in ['province', 'year']]
    for col in numeric_cols:
        if col in analysis_table.columns and analysis_table[col].dtype in ['float64', 'int64']:
            analysis_table[col] = analysis_table[col].round(2)
    
    # Load real annual population from dataset5
    pop_file = Path(output_dir) / 'population_annual_for_analysis.csv'
    if not pop_file.exists():
        raise FileNotFoundError(
            f"Dataset4: {pop_file} not found. Run dataset5_population_etl.py first."
        )
    pop_df = pd.read_csv(pop_file)
    print(f"Dataset4: Loaded {len(pop_df)} population records from dataset5")

    # Keep only static census fields per province (no year dimension yet)
    static_cols = [
        'province', 'land_area_km2', 'average_age', 'median_age',
        'percent_65_plus', 'rural_percent_estimate', 'urban_percent_estimate'
    ]
    static_df = analysis_table[[c for c in static_cols if c in analysis_table.columns]].copy()

    # Merge real annual population onto static province data
    time_series_df = pop_df.merge(static_df, on='province', how='left')

    # Recalculate density per year using real population
    time_series_df['population_density_per_km2'] = (
        time_series_df['population_estimate'] / time_series_df['land_area_km2']
    ).round(2)

    time_series_df = time_series_df.sort_values(['province', 'year']).reset_index(drop=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'demographics_for_analysis.csv'
    time_series_df.to_csv(output_file, index=False)

    print(f"Demographics data cleaned: {len(time_series_df)} records saved to {output_file}")

    return time_series_df

if __name__ == "__main__":
    INPUT_FILES = [
        "../data/raw/dataset4-1_geogrpahic_population.csv",
        "../data/raw/dataset4-2_geogrpahic_population.csv" 
    ]
    OUTPUT_DIR = "../data/processed"
    
    demographics_df = clean_demographics_data(INPUT_FILES, OUTPUT_DIR)