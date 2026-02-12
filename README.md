# Canadian Healthcare Efficiency Analysis

Analyzing healthcare efficiency across **10 Canadian provinces** over **17 years** (2008-2024). Combines ETL data processing, statistical analysis, and machine learning to identify what drives provincial wait time differences and how spending reallocation could improve outcomes.

**Live Dashboard**: [healthcare-efficiency-canada.streamlit.app](https://healthcare-efficiency-canada.streamlit.app)

<img width="708" height="417" alt="image" src="https://github.com/user-attachments/assets/ac9e7b23-c0e8-4b34-b902-95a4c412ed0d" />

## Research Questions

1. **Efficiency Ranking** — Which provinces achieve the best wait times relative to their spending?
2. **Spending Impact** — How do different spending categories (hospitals, physicians, drugs, admin) affect wait times?
3. **Optimization** — What budget-neutral reallocations could reduce wait times?

## Key Findings

- **Ontario** consistently ranks highest in efficiency; **PEI** ranks lowest
- K-means clustering (k=3) identifies 1 High, 5 Medium, 4 Low efficiency tiers
- OLS regression fails to generalize across provinces (GroupKFold R² = -16.5)
- **Fixed Effects OLS** controls for province heterogeneity, achieving R² = 0.345
- Only **physicians** (p=0.0001) and **drugs** (p=0.002) spending remain significant with province effects controlled
- Spending explains ~35% of wait time variance — staffing, scheduling, and patient complexity matter too

## Data Sources

| Source | Description | Records |
|--------|-------------|---------|
| CIHI Wait Times | 50th percentile wait by procedure, province, year | 169 province-years |
| CIHI Spending | 14 spending categories per capita | 169 province-years |
| CIHI Physicians | Physicians per 10k (2017, 2021 snapshots) | 20 data points, extrapolated |
| StatsCan Census | Demographics, land area, urbanization | 10 provinces (static) |
| StatsCan 17-10-0009 | Quarterly population estimates | 170 province-years |

## Project Structure

```
healthcare-efficiency-analysis/
├── app.py                          # Streamlit dashboard (3 pages)
├── .streamlit/config.toml          # Dashboard theme
├── etl/
│   ├── dataset1_wait_times_etl.py        # Volume-weighted wait time aggregation
│   ├── dataset2_total_expenditure_etl.py # 14-category spending breakdown
│   ├── dataset3_physicians_etl.py        # CAGR-extrapolated physician density
│   ├── dataset4_geograpghy_population_etl.py  # Demographics + annual population
│   ├── dataset5_population_etl.py        # StatsCan annual population estimates
│   └── master_dataset_etl.py             # Merge all → 44-column master dataset
├── jupyter/
│   ├── 01_v2_provincial_efficiency.ipynb # K-means tier classification
│   ├── 02_secondary_research_question.ipynb # Spending impact analysis
│   └── 03_v2_resource_optimization.ipynb # Regression diagnostics + model comparison
├── models/                         # Trained model artifacts (.joblib)
├── data/
│   ├── raw/                        # Source CSVs
│   ├── processed/                  # Cleaned per-dataset CSVs
│   └── final/                      # Master analysis dataset
└── requirements.txt
```

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd healthcare-efficiency-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run ETL pipeline (from etl/ directory)
cd etl
python dataset1_wait_times_etl.py
python dataset2_total_expenditure_etl.py
python dataset3_physicians_etl.py
python dataset4_geograpghy_population_etl.py
python dataset5_population_etl.py
python master_dataset_etl.py
cd ..

# Run Streamlit dashboard
streamlit run app.py

# Or run Jupyter notebooks
jupyter notebook
# Execute: 01_v2 → 02 → 03_v2
```

## Model Comparison

| Model | Test R² | 5-Fold CV R² | GroupKFold R² | Notes |
|-------|---------|-------------|---------------|-------|
| OLS (baseline) | 0.108 | 0.138 | -16.5 | Fails cross-province generalization |
| Ridge | 0.111 | 0.008 | -19.2 | Stabilizes coefficients |
| Lasso | 0.110 | 0.009 | -19.3 | No features zeroed out |
| **Fixed Effects OLS** | **0.345** | **0.346** | N/A | **Best model** — controls province heterogeneity |

## ETL Improvements (vs. Original)

| Issue | Before | After |
|-------|--------|-------|
| Population data | Static 2021 census for all years | Real annual estimates from StatsCan |
| Wait time aggregation | Simple mean across procedures | Volume-weighted average |
| Volume parsing | Bug: `"7,700"` silently dropped | Fixed comma parsing |
| Physician extrapolation | Flat (frozen at 2017/2021) | CAGR growth model |
| Tier classification | Arbitrary hardcoded thresholds | K-means clustering with silhouette validation |
| Regression validation | Single 70/30 split, no diagnostics | VIF, residual plots, GroupKFold, assumption tests |

## Limitations

- Age demographics (average_age, percent_65_plus) are static 2021 census values
- Physician density is extrapolated from only 2 data points (2017 and 2021)
- Wait time procedure coverage varies by province (8-12 procedures)
- R² = 0.35 means spending explains ~35% of wait time variance

## Tech Stack

- **ETL**: Python, Pandas, NumPy
- **Analysis**: Scikit-learn, Statsmodels, SciPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Data**: CIHI, Statistics Canada
