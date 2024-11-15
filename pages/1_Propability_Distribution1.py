import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Title and Introduction
st.title("Explore Probability Distributions")
st.write("""
This app demonstrates probability distribution analysis, z-score calculation, and outlier detection.
Data sourced from the EPA Air Quality Index for carbon monoxide across various sites.
""")

# Step 1: Load and Inspect Data
st.header("Step 1: Load and Explore Data")
# Load dataset
data = pd.read_csv("myPortfolio/dataSets/modified_c4_epa_air_quality.csv")  # Ensure correct path
st.write("First 10 rows of data:")
st.write(data.head(10))

# Display data structure
st.write(f"Data Shape (rows, columns): {data.shape}")

# Step 2: Visualize Distribution of AQI Log
st.header("Step 2: Visualize AQI Log Distribution")
# Plot histogram of the existing 'aqi_log' data
fig, ax = plt.subplots()
ax.hist(data["aqi_log"], bins=25, color="skyblue", edgecolor="black")
ax.set_title("Distribution of AQI Log")
ax.set_xlabel("AQI Log")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Calculate mean and standard deviation
mean_aqi_log = data["aqi_log"].mean()
std_aqi_log = data["aqi_log"].std()
st.write(f"Mean of AQI Log: {mean_aqi_log:.2f}")
st.write(f"Standard Deviation of AQI Log: {std_aqi_log:.2f}")

# Step 3: Apply the Empirical Rule
st.header("Step 3: Apply the Empirical Rule")
# Function to calculate empirical rule ranges and percentage
def empirical_rule_check(data, mean, std, multiplier):
    lower_limit = mean - multiplier * std
    upper_limit = mean + multiplier * std
    percent_within = ((data >= lower_limit) & (data <= upper_limit)).mean() * 100
    return lower_limit, upper_limit, percent_within

# Display results for 1, 2, and 3 standard deviations
for i in range(1, 4):
    lower, upper, percent = empirical_rule_check(data['aqi_log'], mean_aqi_log, std_aqi_log, i)
    st.write(f"Within {i} standard deviation(s): {percent:.2f}% of data (Expected: {round(stats.norm.cdf(i) * 2 - 1, 2) * 100}%)")
    st.write(f"Range: {lower:.2f} to {upper:.2f}")

# Step 4: Z-score Calculation and Outlier Detection
st.header("Step 4: Z-score Calculation and Outlier Detection")
# Calculate z-scores and identify outliers
data['z_score'] = stats.zscore(data['aqi_log'])
outliers = data[(data['z_score'] > 3) | (data['z_score'] < -3)]
st.write("Outliers (AQI Log more than 3 standard deviations from the mean):")
st.write(outliers)

# Summary Insights
st.header("Summary Insights")
st.markdown("""
- **Distribution**: The AQI Log data is approximately normal.
- **Empirical Rule**: The data distribution follows the empirical rule closely, with most data points within 1 to 3 standard deviations.
- **Outliers**: High-AQI sites detected as outliers could be prioritized for air quality improvement.
""")
