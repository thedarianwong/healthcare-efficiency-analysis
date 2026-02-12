import pandas as pd
import numpy as np
from pathlib import Path

# Province mapping from abbreviations to full names
PROVINCES = {
    'N.L.': 'Newfoundland and Labrador',
    'P.E.I.': 'Prince Edward Island',
    'N.S.': 'Nova Scotia',
    'N.B.': 'New Brunswick',
    'Que.': 'Quebec',
    'Ont.': 'Ontario',
    'Man.': 'Manitoba',
    'Sask.': 'Saskatchewan',
    'Alta.': 'Alberta',
    'B.C.': 'British Columbia'
}

TARGET_PROVINCES = list(PROVINCES.values())

# Extract province abbreviation from health region name
def extract_province_from_region(region_name):
    region_str = str(region_name).strip()
    
    # Look for province abbreviation in parentheses
    for abb in PROVINCES.keys():
        if f'({abb})' in region_str:
            return PROVINCES[abb]
    
    return None

# Extrapolate and interpolate physician density across 2008-2024
def interpolate_missing_years(physicians_2017, physicians_2021, target_years):
    # CAGR between the two actual data points (annual growth rate)
    if physicians_2017 > 0 and physicians_2021 > 0:
        cagr = (physicians_2021 / physicians_2017) ** (1 / 4) - 1
    else:
        cagr = 0.0

    interpolated_data = []

    for year in target_years:
        if year == 2017:
            density = physicians_2017
            source = 'actual'
        elif year == 2021:
            density = physicians_2021
            source = 'actual'
        elif 2017 < year < 2021:
            # Linear interpolation between the two data points
            progress = (year - 2017) / (2021 - 2017)
            density = physicians_2017 + (physicians_2021 - physicians_2017) * progress
            source = 'interpolated'
        elif year < 2017:
            # CAGR extrapolation backward
            density = physicians_2017 * (1 + cagr) ** (year - 2017)
            source = 'estimated'
        else:
            # CAGR extrapolation forward (year > 2021)
            density = physicians_2021 * (1 + cagr) ** (year - 2021)
            source = 'estimated'

        interpolated_data.append({
            'year': year,
            'physicians_per_10k': round(density, 1),
            'physician_data_source': source
        })

    return interpolated_data

# Clean physician resources data and create analysis table
def clean_physician_data(input_file: str, output_dir: str):
    print("Dataset3: Cleaning physician resources data...")
    df = pd.read_csv(input_file, skiprows=3)

    df.columns = [
        'health_region', 
        'physicians_2017_per_10k', 
        'physicians_2021_per_10k', 
        'provincial_average_2021'
    ]
    
    # Clean the data
    df = df.dropna(subset=['health_region'])  # Remove empty rows
    df['health_region'] = df['health_region'].astype(str).str.strip()
    
    for col in ['physicians_2017_per_10k', 'physicians_2021_per_10k', 'provincial_average_2021']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract province from each health region
    df['province'] = df['health_region'].apply(extract_province_from_region)
    
    # Filter to 10 main provinces only (exclude territories)
    provincial_data = df[df['province'].isin(TARGET_PROVINCES)].copy()
    
    # Group by province and calculate weighted averages
    province_summary = provincial_data.groupby('province').agg({
        'physicians_2017_per_10k': 'mean',
        'physicians_2021_per_10k': 'mean',
        'provincial_average_2021': 'first'
    }).reset_index()
    
    for col in ['physicians_2017_per_10k', 'physicians_2021_per_10k', 'provincial_average_2021']:
        province_summary[col] = province_summary[col].round(1)
    
    # Create time series data (2008-2024) with interpolation
    target_years = list(range(2008, 2025))
    
    all_records = []
    
    for _, province_row in province_summary.iterrows():
        province_name = province_row['province']
        physicians_2017 = province_row['physicians_2017_per_10k']
        physicians_2021 = province_row['physicians_2021_per_10k']
        
        # For missing data
        if pd.isna(physicians_2017) or pd.isna(physicians_2021):
            continue
        
        # Generate interpolated time series
        yearly_data = interpolate_missing_years(physicians_2017, physicians_2021, target_years)
        
        for year_data in yearly_data:
            all_records.append({
                'province': province_name,
                'year': year_data['year'],
                'physicians_per_10k_population': year_data['physicians_per_10k'],
                'physician_data_source': year_data['physician_data_source']
            })
    
    analysis_table = pd.DataFrame(all_records)
    analysis_table = analysis_table.sort_values(['province', 'year']).reset_index(drop=True)
 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'physicians_for_analysis.csv'
    analysis_table.to_csv(output_file, index=False)
    
    print(f"Physicians data cleaned: {len(analysis_table)} records saved to {output_file}")
    
    return analysis_table

if __name__ == "__main__":
    INPUT_FILE = "../data/raw/dataset3_physicians.csv"
    OUTPUT_DIR = "../data/processed"
    
    physicians_df = clean_physician_data(INPUT_FILE, OUTPUT_DIR)