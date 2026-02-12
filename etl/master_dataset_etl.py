import pandas as pd
import numpy as np
from pathlib import Path

# Integrate all 4 cleaned datasets and create strategic efficiency metrics
def create_master_dataset(processed_data_dir="../data/processed", output_dir="../data/final"):
    print("Master Data: Creating master healthcare efficiency dataset...")
    # Load all cleaned datasets
    datasets = {}
    
    datasets['wait_times'] = pd.read_csv(Path(processed_data_dir) / "wait_times_for_analysis.csv")
    datasets['spending'] = pd.read_csv(Path(processed_data_dir) / "spending_breakdown_for_analysis.csv")
    datasets['physicians'] = pd.read_csv(Path(processed_data_dir) / "physicians_for_analysis.csv")
    datasets['demographics'] = pd.read_csv(Path(processed_data_dir) / "demographics_for_analysis.csv")
    
    # Merge all datasets
    master_df = datasets['wait_times'].copy()
    master_df = master_df.merge(datasets['spending'], on=['province', 'year'], how='left')
    master_df = master_df.merge(datasets['physicians'], on=['province', 'year'], how='left')
    master_df = master_df.merge(datasets['demographics'], on=['province', 'year'], how='left')
    
    # Primary efficiency metrics
    master_df['raw_efficiency'] = master_df['total_spending_per_capita'] / master_df['avg_wait_time']
    master_df['geographic_challenge'] = 1 / np.log(master_df['population_density_per_km2'] + 1)
    best_efficiency = master_df['raw_efficiency'].max()
    master_df['efficiency_gap'] = best_efficiency - master_df['raw_efficiency']
    
    # Resource allocation efficiency metrics
    master_df['hospitals_efficiency'] = master_df['hospitals'] / master_df['avg_wait_time']
    master_df['physicians_efficiency'] = master_df['physicians'] / master_df['avg_wait_time']
    master_df['drugs_efficiency'] = master_df['drugs'] / master_df['avg_wait_time']
    master_df['admin_efficiency'] = master_df['administration'] / master_df['avg_wait_time']
    master_df['other_professionals_efficiency'] = master_df['other_professionals'] / master_df['avg_wait_time']
    master_df['public_health_efficiency'] = master_df['public_health'] / master_df['avg_wait_time']
    
    # Strategic allocation ratios
    master_df['admin_overhead_pct'] = (master_df['administration'] / master_df['total_spending_per_capita']) * 100
    master_df['clinical_focus_pct'] = ((master_df['hospitals'] + master_df['physicians']) / master_df['total_spending_per_capita']) * 100

    # COVID period flag
    master_df['covid_period'] = master_df['year'].isin([2020, 2021])

    # Summary
    print(f"Master dataset created: {len(master_df)} records")
    print(f"Provinces: {master_df['province'].nunique()}")
    print(f"Years: {master_df['year'].min()}-{master_df['year'].max()}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'healthcare_efficiency_master.csv'
    master_df.to_csv(output_file, index=False)
    
    print(f"Master dataset created: {len(master_df)} records saved to {output_file}")
    
    return master_df
    


if __name__ == "__main__":
    master_data = create_master_dataset()