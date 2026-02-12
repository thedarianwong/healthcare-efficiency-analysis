import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Data & model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/final/healthcare_efficiency_master.csv")

@st.cache_resource
def load_models():
    models_dir = Path("models")
    return {
        "ols": joblib.load(models_dir / "ols_baseline.joblib"),
        "ridge": joblib.load(models_dir / "ridge.joblib"),
        "lasso": joblib.load(models_dir / "lasso.joblib"),
        "fe_ols": joblib.load(models_dir / "fixed_effects_ols.joblib"),
        "scaler": joblib.load(models_dir / "standard_scaler.joblib"),
        "meta": joblib.load(models_dir / "model_metadata.joblib"),
    }

df = load_data()
models = load_models()
meta = models["meta"]
SPENDING_CATS = meta["spending_categories"]
PROVINCE_DUMMY_COLS = meta["province_dummies_columns"]
PROVINCES = sorted(df["province"].unique())

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Healthcare Efficiency")
page = st.sidebar.radio("Navigate", ["Overview", "Provincial Explorer", "Policy Simulator"])

# ===========================================================================
# PAGE 1: Overview
# ===========================================================================

if page == "Overview":
    st.title("Canadian Healthcare Efficiency Analysis")
    st.markdown(
        "Analyzing healthcare efficiency across **10 Canadian provinces** over "
        "**17 years** (2008-2024). Combines ETL processing, statistical analysis, "
        "and policy scenario modeling."
    )

    st.subheader("Key Findings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Province", "Ontario", "R\u00b2 = 299.1 efficiency score")
    with col2:
        st.metric("Best Model", "Fixed Effects OLS", f"R\u00b2 = {meta['best_test_r2']:.3f}")
    with col3:
        st.metric("Records", f"{len(df)}", "10 provinces x 17 years")

    st.subheader("Data Pipeline")
    st.code(
        "data/raw/ (5 source files)\n"
        "  -> etl/dataset[1-5]_*_etl.py\n"
        "    -> data/processed/ (5 cleaned CSVs)\n"
        "      -> etl/master_dataset_etl.py\n"
        "        -> data/final/healthcare_efficiency_master.csv\n"
        "          -> jupyter/01_v2, 03_v2 notebooks\n"
        "            -> models/ (4 trained models)\n"
        "              -> app.py (this dashboard)",
        language=None,
    )

    st.subheader("Model Comparison")
    comparison_data = {
        "Model": ["OLS (baseline)", "Ridge", "Lasso", "Fixed Effects OLS"],
        "Test R\u00b2": [0.108, 0.111, 0.110, 0.345],
        "5-Fold CV R\u00b2": [0.138, 0.008, 0.009, 0.346],
        "Key Insight": [
            "Fails cross-province generalization",
            "Stabilizes coefficients, still fails cross-province",
            "No features zeroed out",
            "Controls province heterogeneity (best model)",
        ],
    }
    st.dataframe(pd.DataFrame(comparison_data), width='stretch', hide_index=True)

    st.subheader("Limitations")
    st.markdown(
        "- Age demographics (average_age, percent_65_plus) are static 2021 census values\n"
        "- Physician density is interpolated/extrapolated from only 2 data points (2017, 2021)\n"
        "- Wait time procedure coverage varies by province (8-12 procedures)\n"
        "- R\u00b2 = 0.35 means spending explains ~35% of wait time variance; "
        "staffing, scheduling, and patient complexity matter too"
    )

# ===========================================================================
# PAGE 2: Provincial Explorer
# ===========================================================================

elif page == "Provincial Explorer":
    st.title("Provincial Explorer")

    # Controls
    col_a, col_b = st.columns([1, 2])
    with col_a:
        selected_province = st.selectbox("Province", PROVINCES)
    with col_b:
        year_range = st.slider(
            "Year Range",
            int(df["year"].min()),
            int(df["year"].max()),
            (int(df["year"].min()), int(df["year"].max())),
        )

    filtered = df[
        (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    ].copy()

    # --- Efficiency rankings bar chart with K-means tiers ---
    st.subheader("Efficiency Rankings")

    prov_avg = (
        filtered.groupby("province")
        .agg({"raw_efficiency": "mean", "avg_wait_time": "mean", "total_spending_per_capita": "mean"})
        .round(1)
        .reset_index()
        .sort_values("raw_efficiency", ascending=False)
    )

    # K-means tiers (recompute on filtered data)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    cluster_cols = ["raw_efficiency", "avg_wait_time", "total_spending_per_capita"]
    X_cl = StandardScaler().fit_transform(prov_avg[cluster_cols])
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_cl)
    prov_avg["cluster"] = km.labels_
    cluster_means = prov_avg.groupby("cluster")["raw_efficiency"].mean().sort_values(ascending=False)
    tier_map = {cluster_means.index[0]: "High", cluster_means.index[1]: "Medium", cluster_means.index[2]: "Low"}
    prov_avg["tier"] = prov_avg["cluster"].map(tier_map)

    color_map = {"High": "#2E8B57", "Medium": "#FFD700", "Low": "#DC143C"}

    fig_rank = px.bar(
        prov_avg,
        x="province",
        y="raw_efficiency",
        color="tier",
        color_discrete_map=color_map,
        labels={"raw_efficiency": "Efficiency Score", "province": "Province", "tier": "Tier"},
        title="Provincial Efficiency Rankings (K-Means Tiers)",
    )
    national_avg = prov_avg["raw_efficiency"].mean()
    fig_rank.add_hline(y=national_avg, line_dash="dash", line_color="gray",
                       annotation_text=f"National Avg: {national_avg:.1f}")
    st.plotly_chart(fig_rank, width='stretch')

    # --- Province detail ---
    st.subheader(f"{selected_province} Detail")
    prov_data = filtered[filtered["province"] == selected_province].sort_values("year")

    if len(prov_data) == 0:
        st.warning("No data for this province in the selected year range.")
    else:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Wait Time", f"{prov_data['avg_wait_time'].mean():.1f} days")
        k2.metric("Avg Spending/Capita", f"${prov_data['total_spending_per_capita'].mean():,.0f}")
        k3.metric("Avg Efficiency", f"{prov_data['raw_efficiency'].mean():.1f}")
        k4.metric("Population (latest)", f"{prov_data['population_estimate'].iloc[-1]:,.0f}")

        # Wait time trend
        fig_wait = px.line(
            prov_data, x="year", y="avg_wait_time",
            title=f"{selected_province} — Wait Time Trend",
            labels={"avg_wait_time": "Avg Wait Time (days)", "year": "Year"},
        )
        # Highlight COVID period
        fig_wait.add_vrect(x0=2020, x1=2021, fillcolor="red", opacity=0.1,
                           annotation_text="COVID", annotation_position="top left")
        st.plotly_chart(fig_wait, width='stretch')

        # Spending breakdown
        spending_long = prov_data.melt(
            id_vars=["year"],
            value_vars=SPENDING_CATS,
            var_name="Category",
            value_name="Spending ($)",
        )
        fig_spend = px.area(
            spending_long, x="year", y="Spending ($)", color="Category",
            title=f"{selected_province} — Spending Breakdown",
        )
        st.plotly_chart(fig_spend, width='stretch')

# ===========================================================================
# PAGE 3: Policy Simulator
# ===========================================================================

elif page == "Policy Simulator":
    st.title("Policy Simulator")
    st.markdown(
        "Adjust spending category sliders to simulate budget-neutral reallocations. "
        "The **Fixed Effects OLS** model predicts the resulting wait time change."
    )

    # Province selector for fixed effects
    sim_province = st.selectbox("Province", PROVINCES, key="sim_prov")

    # Current averages for this province
    prov_means = df[df["province"] == sim_province][SPENDING_CATS].mean()
    current_total = prov_means.sum()

    st.subheader("Spending Sliders (per capita $)")
    st.caption(f"Current total: ${current_total:,.0f} — adjust categories, budget-neutral constraint shown below.")

    slider_vals = {}
    cols = st.columns(3)
    for i, cat in enumerate(SPENDING_CATS):
        with cols[i % 3]:
            mn = float(df[cat].min()) * 0.5
            mx = float(df[cat].max()) * 1.5
            slider_vals[cat] = st.slider(
                cat.replace("_", " ").title(),
                min_value=int(mn),
                max_value=int(mx),
                value=int(prov_means[cat]),
                step=10,
                key=f"slider_{cat}",
            )

    new_total = sum(slider_vals.values())
    budget_diff = new_total - current_total

    if abs(budget_diff) < 1:
        st.success(f"Budget neutral (${new_total:,.0f})")
    elif budget_diff > 0:
        st.warning(f"Over budget by ${budget_diff:,.0f} (${new_total:,.0f} vs ${current_total:,.0f})")
    else:
        st.info(f"Under budget by ${abs(budget_diff):,.0f} (${new_total:,.0f} vs ${current_total:,.0f})")

    # Build feature vector for fixed effects model
    spending_vec = [slider_vals[c] for c in SPENDING_CATS]
    province_vec = [1.0 if p == sim_province else 0.0 for p in PROVINCE_DUMMY_COLS]
    feature_cols = SPENDING_CATS + PROVINCE_DUMMY_COLS
    feature_vec = pd.DataFrame([spending_vec + province_vec], columns=feature_cols)

    # Baseline prediction (current spending)
    baseline_spending = [prov_means[c] for c in SPENDING_CATS]
    baseline_vec = pd.DataFrame([baseline_spending + province_vec], columns=feature_cols)

    fe_model = models["fe_ols"]
    baseline_pred = fe_model.predict(baseline_vec)[0]
    new_pred = fe_model.predict(feature_vec)[0]
    change = new_pred - baseline_pred

    st.subheader("Prediction")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Current Predicted Wait", f"{baseline_pred:.1f} days")
    pc2.metric("New Predicted Wait", f"{new_pred:.1f} days")
    pc3.metric("Change", f"{change:+.1f} days", delta_color="inverse")

    # Extrapolation warning
    training_ranges = {cat: (df[cat].min(), df[cat].max()) for cat in SPENDING_CATS}
    extrapolating = []
    for cat in SPENDING_CATS:
        lo, hi = training_ranges[cat]
        if slider_vals[cat] < lo or slider_vals[cat] > hi:
            extrapolating.append(cat)

    if extrapolating:
        st.warning(
            f"Values for {', '.join(extrapolating)} are outside the training data range. "
            "Predictions may be unreliable."
        )

    st.caption(
        f"Model: Fixed Effects OLS | Test R\u00b2 = {meta['best_test_r2']:.3f} | "
        f"5-Fold CV R\u00b2 = {meta['best_cv_r2']:.3f} | "
        "Spending explains ~35% of wait time variance."
    )
