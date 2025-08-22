# 🏥 Canadian Healthcare Efficiency Analysis

> **Advanced data science analysis revealing provincial healthcare efficiency patterns and optimization strategies across Canada**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![Data Science](https://img.shields.io/badge/Domain-Healthcare%20Analytics-green.svg)](https://github.com/dtw11/healthcare-efficiency-canada)

<img width="1920" height="1080" alt="Canadian Healthcare Efficiency Banner" src="https://github.com/user-attachments/assets/54139199-5038-4325-9f9d-df16f8db34b9" />

## 🎯 Project Overview

This comprehensive data science project analyzes healthcare efficiency across 10 Canadian provinces over 17 years (2008-2024), combining machine learning, statistical analysis, and optimization modeling to identify actionable insights for healthcare policy improvement.

**Key Achievement**: Developed linear regression model achieving statistical significance (p<0.001) and identified key spending relationships that could optimize provincial healthcare efficiency.

## 📊 Data & Methodology

- **Scale**: 169 records × 41 features spanning demographics, spending, resources, and outcomes
- **Sources**: Canadian Institute for Health Information (CIHI), Statistics Canada
- **Techniques**: Statistical hypothesis testing, linear regression, correlation analysis, ANOVA testing
- **Geographic Controls**: Population density, rural/urban distribution, land area adjustments

## 🔬 Research Questions

1. **Efficiency Ranking**: Which provinces achieve optimal wait times relative to resource investment?
2. **Resource Strategy**: How do spending patterns (hospitals vs physicians vs administration) impact performance?
3. **Optimization**: What reallocation strategies could minimize national wait times?

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+ | Git | Jupyter Notebook
```

### Installation & Setup
```bash
# Clone and setup
git clone https://github.com/your-username/healthcare-efficiency-canada.git
cd healthcare-efficiency-canada

# Environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Analysis Pipeline
```bash
# Navigate to ETL directory
cd etl

# Run individual dataset processing
python dataset1_wait_times_etl.py
python dataset2_total_expenditure_etl.py
python dataset3_physicians_etl.py
python dataset4_geograpghy_population_etl.py

# Verify processed datasets exist
ls ../data/processed/

# Run master dataset integration
python master_dataset_etl.py

# Start Jupyter
jupyter notebook

# Execute notebooks in sequence:
#    → jupyter/01_primary_research_question.ipynb
#    → jupyter/02_secondary_research_question.ipynb
#    → jupyter/03_tertiary_research_question.ipynb
```

## 📈 Key Findings

- **Top Performers**: Ontario and Newfoundland & Labrador lead with 189.6 average efficiency score and 44.6-day wait times
- **Efficiency Gap**: 91-point efficiency range between highest (Ontario: 189.8) and lowest (PEI: 98.8) performers
- **Administrative Impact**: Each $100 increase in administrative spending correlates with 14.2-day longer wait times (p<0.001)
- **Physician Investment**: Most cost-effective lever with 11.4-day wait time reduction per $100 investment
- **Overhead Insight**: High-efficiency provinces maintain 0.6% lower administrative overhead (2.6% vs 3.2%)

## 🛠️ Technical Stack

- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib
- **ML Models**: Linear Regression for interpretable coefficients
- **Statistical Analysis**: SciPy, Statsmodels, ANOVA, T-tests, Correlation Analysis

## 📁 Project Structure

```
healthcare-efficiency-canada/
├── data/
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned individual datasets
│   └── final/         # Master analysis dataset
├── etl/               # Data processing scripts
├── jupyter/           # Analysis notebooks
└── requirements.txt   # Dependencies
```

## 📊 Data Visualizations

The project includes dashboards and comprehensive visualizations showing:
- Provincial efficiency rankings bar chart with tier color-coding
- Resource allocation patterns across efficiency tiers and administrative overhead correlation with
provincial efficiency scores
- Machine learning model predictions of wait time impacts from $100 spending changes and policy
scenario outcomes for resource reallocation strategies

---

⭐ **Star this repo if you find it useful for healthcare analytics or data science portfolio examples!**
