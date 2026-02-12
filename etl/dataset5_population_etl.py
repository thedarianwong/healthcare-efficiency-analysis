import pandas as pd
import numpy as np
from pathlib import Path

# 10 Canadian provinces (exclude territories)
TARGET_PROVINCES = [
    'Newfoundland and Labrador',
    'Prince Edward Island',
    'Nova Scotia',
    'New Brunswick',
    'Quebec',
    'Ontario',
    'Manitoba',
    'Saskatchewan',
    'Alberta',
    'British Columbia'
]

YEAR_RANGE = range(2008, 2025)  # 2008-2024 inclusive


def clean_population_data(input_file: str, output_dir: str) -> pd.DataFrame:
    """Parse StatsCan Table 17-10-0009-01 quarterly population estimates.

    Uses Q3 (July 1) estimates as the annual reference point,
    which is the standard StatsCan convention for annual population.
    """
    print("Dataset5: Cleaning population estimates data...")

    df = pd.read_csv(input_file)

    # Filter to 10 provinces only
    df = df[df['GEO'].isin(TARGET_PROVINCES)].copy()

    # Parse REF_DATE into year and quarter
    df['year'] = df['REF_DATE'].str[:4].astype(int)
    df['quarter'] = df['REF_DATE'].str[5:7].astype(int)  # 01=Q1, 04=Q2, 07=Q3, 10=Q4

    # Use Q3 (July 1) as the annual population estimate
    annual = df[df['quarter'] == 7].copy()

    # Filter to target year range
    annual = annual[annual['year'].isin(YEAR_RANGE)].copy()

    # Keep relevant columns and rename
    annual = annual[['GEO', 'year', 'VALUE']].rename(columns={
        'GEO': 'province',
        'VALUE': 'population_estimate'
    })

    annual['population_estimate'] = pd.to_numeric(
        annual['population_estimate'], errors='coerce'
    )

    # Sort for consistent output
    annual = annual.sort_values(['province', 'year']).reset_index(drop=True)

    # Calculate year-over-year growth rate
    annual['population_growth_rate'] = (
        annual.groupby('province')['population_estimate']
        .pct_change() * 100
    ).round(2)

    # Validate
    expected_records = len(TARGET_PROVINCES) * len(YEAR_RANGE)
    actual_records = len(annual)
    print(f"Dataset5: Expected {expected_records} records, got {actual_records}")

    missing = annual[annual['population_estimate'].isna()]
    if len(missing) > 0:
        print(f"Dataset5: WARNING — {len(missing)} records with missing population")

    # Spot-check: Ontario 2021 should be ~14.2M (census was 14,223,942)
    ont_2021 = annual[
        (annual['province'] == 'Ontario') & (annual['year'] == 2021)
    ]['population_estimate'].values
    if len(ont_2021) > 0:
        print(f"Dataset5: Spot-check Ontario 2021 = {ont_2021[0]:,.0f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'population_annual_for_analysis.csv'
    annual.to_csv(output_file, index=False)

    print(f"Dataset5: {len(annual)} records saved to {output_file}")
    return annual


if __name__ == "__main__":
    INPUT_FILE = "../data/raw/17100009.csv"
    OUTPUT_DIR = "../data/processed"

    population_df = clean_population_data(INPUT_FILE, OUTPUT_DIR)
