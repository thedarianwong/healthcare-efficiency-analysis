# Canadian Healthcare Efficiency Analysis

> **Data-driven analysis of provincial healthcare efficiency and resource allocation patterns across Canada (2008-2024)**

## Introduction

This project analyzes healthcare efficiency across 10 Canadian provinces over 17 years, integrating multiple CIHI and StatsCan datasets to identify resource allocation strategies and improvement opportunities for reducing wait times in priority medical procedures.

**Data Sources**: CIHI wait times, CIHI healthcare spending, CIHI physician resources and Statistics Canada demographics.

## Research Questions

1. **Primary**: Which provinces deliver the most efficient healthcare performance that achieves shorter wait times relative to resource investments?

2. **Secondary**: What resource allocation strategies (hospital vs physician vs administrative spending) distinguish efficient from inefficient provinces?

3. **Tertiary**: What resource allocation strategies would optimize wait time reduction through budget relocation?

## 🔧 Prerequisites

- Python 3.8+
- Git

## 🚀 How to Run

### 1. Setup Environment
```bash
# Clone repository
git clone git@github.sfu.ca:dtw11/healthcare-efficiency-canada.git
cd healthcare-efficiency-canada

# Create virtual environment
python -m venv healthcare_venv
source healthcare_venv/bin/activate  # On Windows: healthcare_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run ETL Pipeline
**Execute scripts in this exact order:**

```bash
# Navigate to ETL directory
cd etl

# Run individual dataset processing
python dataset1_wait_times_etl.py
python dataset2_total_expenditure_etl.py
python dataset3_physicians_etl.py
python dataset4_geogrpahy_population_etl.py

# Verify processed datasets exist
ls ../data/processed/

# Run master dataset integration
python master_dataset_etl.py
```

**Success Check**: Verify `data/final/healthcare_efficiency_master.csv` exists

## 📊 Analysis in Jupyter

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 1. jupyter/01_primary_research_question.ipynb  - Provincial efficiency rankings
# 2. jupyter/02_secondary_research_question.ipynb  - Resource allocation analysis  
# 3. jupyter/03_tertiary_research_question.ipynb  - Predictive modeling
```
