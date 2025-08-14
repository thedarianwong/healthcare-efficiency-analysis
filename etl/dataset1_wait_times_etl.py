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

# Clean wait times data and create analysis table
def clean_wait_times_data(input_file: str, output_dir: str):
    print("Dataset1: Cleaning wait times data...")
    
    df = pd.read_csv(input_file, low_memory=False)
    
    expected_columns = [
        'reporting_level', 'province', 'region', 'procedure', 
        'metric', 'year', 'unit', 'value'
    ]
    
    if len(df.columns) >= 8:
        df = df.iloc[:, :8].copy()
        df.columns = expected_columns
    else:
        df.columns = expected_columns[:len(df.columns)]
    
    # Filter to Provincial data only
    df = df[df['reporting_level'] == 'Provincial'].copy()
    df['province'] = df['province'].str.strip()
    df = df[df['province'].isin(PROVINCES.keys())].copy()
    
    # Clean years - numeric years only
    df['year_clean'] = df['year'].astype(str).str.strip()
    year_pattern = df['year_clean'].str.match(r'^\d{4}$')
    df = df[year_pattern].copy()
    df['year'] = pd.to_numeric(df['year_clean'])
    df = df[(df['year'] >= 2008) & (df['year'] <= 2024)].copy()
    
    # Filter to relevant metrics
    wait_times = df[
        (df['metric'] == '50th Percentile') & 
        (df['unit'] == 'Days')
    ].copy()
    
    benchmarks = df[df['metric'] == '% Meeting Benchmark'].copy()
    volumes = df[
        (df['metric'] == 'Volume') &
        (df['unit'] == 'Number of cases')
    ].copy()
    
    # Clean wait times
    wait_times['value'] = wait_times['value'].astype(str).str.strip()
    wait_times.loc[wait_times['value'].isin(['n/a', 'N/A', '', 'nan', 'Days']), 'value'] = np.nan
    wait_times['wait_time'] = pd.to_numeric(wait_times['value'], errors='coerce')
    wait_times = wait_times[
        (wait_times['wait_time'].notna()) & 
        (wait_times['wait_time'] > 0) & 
        (wait_times['wait_time'] <= 500)
    ].copy()
    
    # Clean benchmark data
    benchmarks['value'] = benchmarks['value'].astype(str).str.strip()
    benchmarks.loc[benchmarks['value'].isin(['n/a', 'N/A', '', 'nan']), 'value'] = np.nan
    benchmarks['benchmark_pct'] = pd.to_numeric(benchmarks['value'], errors='coerce')
    benchmarks = benchmarks[
        (benchmarks['benchmark_pct'].notna()) & 
        (benchmarks['benchmark_pct'] >= 0) & 
        (benchmarks['benchmark_pct'] <= 100)
    ].copy()
    
    # Clean volume data
    volumes['value'] = volumes['value'].astype(str).str.strip()
    missing_values = ['n/a', 'N/A', 'na', 'NA', '', 'nan', 'NaN', 'None', 'none']
    volumes.loc[volumes['value'].isin(missing_values), 'value'] = np.nan
    volumes['volume_cases'] = pd.to_numeric(volumes['value'], errors='coerce')
    volumes = volumes[volumes['volume_cases'].notna()].copy()
    volumes = volumes[volumes['volume_cases'] >= 0].copy()
    
    # Create summaries by province-year
    wait_summary = wait_times.groupby(['province', 'year']).agg({
        'wait_time': 'mean',
        'procedure': 'count'
    }).round(1).reset_index()
    wait_summary.columns = ['province', 'year', 'avg_wait_time', 'procedures_with_wait_times']
    
    benchmark_summary = benchmarks.groupby(['province', 'year']).agg({
        'benchmark_pct': 'mean'
    }).round(1).reset_index()
    
    # Merge wait times and benchmarks
    analysis_table = wait_summary.merge(
        benchmark_summary, 
        on=['province', 'year'], 
        how='left'
    )
    
    analysis_table['benchmark_pct'] = analysis_table['benchmark_pct'].fillna(np.nan)
    
    # Save output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'wait_times_for_analysis.csv'
    analysis_table.to_csv(output_file, index=False)
    
    print(f"Wait times cleaned: {len(analysis_table)} records saved to {output_file}")
    
    return analysis_table

if __name__ == "__main__":
    INPUT_FILE = "../data/raw/dataset1_wait_times.csv"
    OUTPUT_DIR = "../data/processed"
    
    wait_times_df = clean_wait_times_data(INPUT_FILE, OUTPUT_DIR)