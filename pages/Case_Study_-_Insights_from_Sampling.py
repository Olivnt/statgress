import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Data (First 100 Rows)", "View Full Dataset"])

# Load the dataset (ensure the file path is correct)
@st.cache_data
def load_data():
    return pd.read_csv("myPortfolio/dataSets/c4_epa_air_quality.csv")

data = load_data()

# Main Page
if page == "Main Page":
    # Title and description
    st.title("Understanding Data Distributions Through Sampling")
    st.markdown("""
    This analysis explores how data distributions behave through sampling, illustrating the principles of averages and patterns using air quality data.
    """)

    # Context about the dataset and concept
    st.header("About This Analysis")
    st.write("""
    This analysis investigates the behavior of data distributions through sampling, illustrating the principles of averages and patterns using air quality data.
    """)

    # Calculate population statistics
    population_mean = data['aqi'].mean()
    population_std = data['aqi'].std()
    st.write(f"**Population Mean (μ):** {population_mean:.2f}")
    st.write(f"**Population Standard Deviation (σ):** {population_std:.2f}")

    # Input for number of samples
    n_samples = st.slider("Select the sample size (n):", min_value=10, max_value=1000, value=50)
    n_repeats = st.slider("Select the number of repeated samples:", min_value=100, max_value=10000, value=1000)

    # Draw repeated samples and calculate sample means
    sample_means = [data['aqi'].sample(n_samples, replace=True).mean() for _ in range(n_repeats)]
    mean_sample_means = np.mean(sample_means)
    standard_error = population_std / np.sqrt(n_samples)
    st.write(f"**Mean of Sample Means:** {mean_sample_means:.2f}")
    st.write(f"**Standard Error (σ/√n):** {standard_error:.2f}")

    # Plotting the sampling distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_means, bins=25, density=True, alpha=0.4, label="Sample Means Histogram")
    xmin, xmax = ax.set_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, population_mean, standard_error)
    ax.plot(x, p, 'k', linewidth=2, label="Normal Curve (CLT)")
    ax.axvline(population_mean, color='m', linestyle='solid', label="Population Mean")
    ax.axvline(mean_sample_means, color='b', linestyle=':', label="Mean of Sample Means")
    ax.set_title("Sampling Distribution of the Sample Mean")
    ax.set_xlabel("Sample Mean of AQI")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # Insights and takeaways
    st.header("Key Insights")
    st.markdown("""
    - **Normal Distribution:** The histogram of the sampling distribution resembles a normal curve.
    - **Mean of Sample Means:** The mean of sample means closely approximates the population mean (μ).
    - **Standard Error:** The standard error (σ/√n) quantifies the variability of sample means and decreases as the sample size increases.
    """)

    # Additional applications and concluding remarks
    st.header("Applications of Sampling Distributions")
    st.markdown("""
    Sampling distributions are critical in data science and statistics. They are used for:
    - **Hypothesis Testing:** Drawing inferences about population parameters.
    - **Confidence Intervals:** Estimating population parameters with defined precision.
    - **Predictive Modeling:** Leveraging inferential techniques for real-world predictions.
    """)

# View Data (First 100 Rows) Page
elif page == "View Data (First 100 Rows)":
    st.title("View Dataset (First 100 Rows)")
    st.write("""
    This page displays the first 100 rows of the air quality dataset used in the analysis.
    """)
    st.dataframe(data.head(100))

# View Full Dataset Page
elif page == "View Full Dataset":
    st.title("View Full Dataset")
    st.write("""
    This page provides the complete dataset used in the analysis.
    """)
    st.dataframe(data)
