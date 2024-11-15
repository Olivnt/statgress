import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Title and description
st.title("Exploring Sampling and the Central Limit Theorem")
st.write("""
This Streamlit app demonstrates sampling distribution of sample means 
and the central limit theorem using air quality data.
""")

# Load the dataset directly (ensure correct path to 'aqi' column)
data = pd.read_csv("myPortfolio/dataSets/c4_epa_air_quality.csv")  # Update file path as needed
st.write("Loaded data:")
st.write(data.head())

# Calculate sample and population statistics using 'aqi'
population_mean = data['aqi'].mean()
population_std = data['aqi'].std()
st.write(f"Population mean (μ) of AQI: {population_mean:.2f}")
st.write(f"Population standard deviation (σ) of AQI: {population_std:.2f}")

# Input: Number of samples
n_samples = st.slider("Number of samples to draw (n):", min_value=10, max_value=1000, value=50)
n_repeats = st.slider("Number of repeated samples:", min_value=100, max_value=10000, value=1000)

# Draw repeated samples and calculate sample means
sample_means = [data['aqi'].sample(n_samples, replace=True).mean() for _ in range(n_repeats)]
mean_sample_means = np.mean(sample_means)
standard_error = population_std / np.sqrt(n_samples)
st.write(f"Mean of sample means: {mean_sample_means:.2f}")
st.write(f"Standard error (σ/√n): {standard_error:.2f}")

# Plotting the sampling distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sample_means, bins=25, density=True, alpha=0.4, label="Sample means histogram")
xmin, xmax = ax.set_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, population_mean, standard_error)
ax.plot(x, p, 'k', linewidth=2, label="Normal curve (CLT)")
ax.axvline(population_mean, color='m', linestyle='solid', label="Population mean")
ax.axvline(mean_sample_means, color='b', linestyle=':', label="Mean of sample means")
ax.set_title("Sampling distribution of the sample mean")
ax.set_xlabel("Sample mean of AQI")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

# Insights and takeaways
st.write("## Insights")
st.markdown("""
- The histogram of the sampling distribution approximates a normal distribution, as described by the central limit theorem.
- The mean of sample means is close to the population mean.
- The standard error (σ/√n) describes the spread of the sample means around the population mean.
""")
