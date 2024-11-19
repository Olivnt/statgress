import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Raw Data"])

# Load dataset (ensure correct file path)
@st.cache_data
def load_data():
    return pd.read_csv("dataSets/modified_c4_epa_air_quality.csv")

data = load_data()

# Main Page
if page == "Main Page":
    # Title and Introduction
    st.title("Exploring Data Distributions")
    st.markdown("""
    This project examines probability distributions, highlights the role of z-scores in detecting anomalies, 
    and uses air quality data to uncover patterns and outliers.
    """)

    # Step 1: Load and Inspect Data
    st.header("Step 1: Load and Explore Data")
    st.write("### First 10 Rows of the Dataset")
    st.write(data.head(10))
    st.write(f"**Data Shape (rows, columns):** {data.shape}")

    # Step 2: Visualize Distribution of AQI Log
    st.header("Step 2: Visualize AQI Log Distribution")
    fig, ax = plt.subplots()
    ax.hist(data["aqi_log"], bins=25, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of AQI Log")
    ax.set_xlabel("AQI Log")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Calculate mean and standard deviation
    mean_aqi_log = data["aqi_log"].mean()
    std_aqi_log = data["aqi_log"].std()
    st.write(f"**Mean of AQI Log:** {mean_aqi_log:.2f}")
    st.write(f"**Standard Deviation of AQI Log:** {std_aqi_log:.2f}")

    # Step 3: Apply the Empirical Rule
    st.header("Step 3: Apply the Empirical Rule")
    st.write("### Empirical Rule Analysis")
    for i in range(1, 4):
        lower, upper, percent = (
            mean_aqi_log - i * std_aqi_log,
            mean_aqi_log + i * std_aqi_log,
            ((data["aqi_log"] >= mean_aqi_log - i * std_aqi_log) & (data["aqi_log"] <= mean_aqi_log + i * std_aqi_log)).mean() * 100,
        )
        st.write(f"- **Within {i} standard deviation(s):** {percent:.2f}% of data (Expected: {round(stats.norm.cdf(i) * 2 - 1, 2) * 100}%)")
        st.write(f"  **Range:** {lower:.2f} to {upper:.2f}")

    # Step 4: Z-score Calculation and Outlier Detection
    st.header("Step 4: Z-score Calculation and Outlier Detection")
    data['z_score'] = stats.zscore(data['aqi_log'])
    outliers = data[(data['z_score'] > 3) | (data['z_score'] < -3)]
    st.write("### Outliers Detected")
    st.write("Outliers (AQI Log more than 3 standard deviations from the mean):")
    st.write(outliers)

    # Summary Insights
    st.header("Summary Insights")
    st.markdown("""
    - **Distribution:** The AQI Log data appears approximately normal.  
    - **Empirical Rule:** The distribution aligns with the empirical rule, with most data points falling within 1 to 3 standard deviations.  
    - **Outliers:** Sites with unusually high or low AQI Log values (detected as outliers) may warrant further investigation or targeted air quality improvement measures.
    """)

# View Raw Data Page
elif page == "View Raw Data":
    st.title("View Raw Dataset")
    st.markdown("""
    This page provides a view of the full raw dataset used in the analysis.  
    You can inspect the data structure and values in detail.
    """)
    st.dataframe(data)
