import streamlit as st
import pandas as pd
from scipy import stats

st.title("TikTok Project: Statistical Analysis of View Counts by Account Verification Status")

# Part 1: Imports and Data Loading
st.header("Data Loading and Preparation")

# Load the dataset
@st.cache_data  # Cache the data to enhance performance
def load_data():
    return pd.read_csv("myPortfolio/dataSets/tiktok_dataset.csv")

# Load and display data
data = load_data()
st.write("Loaded Dataset:", data.head())

# Part 2: Data Cleaning
st.header("Data Cleaning")

# Check for and handle missing values
missing_values = data.isna().sum()
st.write("Missing Values in Each Column:", missing_values)

# Drop rows with missing values
data = data.dropna()
st.write("Dataset after dropping missing values:", data.head())

# Part 3: Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis")

# Display descriptive statistics for the dataset
st.subheader("Descriptive Statistics")
st.write(data.describe())

# Calculate and display the average view count for each verification status
st.subheader("Average View Count by Verification Status")
average_views = data.groupby("verified_status")["video_view_count"].mean()
st.write(average_views)

# Focus on Verified and Not Verified groups for hypothesis testing
not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]

# Part 4: Hypothesis Testing
st.header("Hypothesis Testing")

# Define the hypotheses
st.write("**Hypotheses**")
st.write("**Null Hypothesis (H₀):** There is no difference in average view count between TikTok videos posted by verified and unverified accounts.")
st.write("**Alternative Hypothesis (H₁):** There is a difference in average view count between TikTok videos posted by verified and unverified accounts.")

# Conduct a two-sample t-test
t_test_result = stats.ttest_ind(a=not_verified, b=verified, equal_var=False)

# Display the t-test results
st.subheader("T-test Results")
st.write(f"Statistic: {t_test_result.statistic}")
st.write(f"P-value: {t_test_result.pvalue}")

# Interpret the p-value
alpha = 0.05  # Significance level of 5%
if t_test_result.pvalue < alpha:
    st.write("**Conclusion:** Reject the null hypothesis. There is a statistically significant difference in the average view count between verified and unverified accounts.")
else:
    st.write("**Conclusion:** Fail to reject the null hypothesis. There is no statistically significant difference in the average view count between verified and unverified accounts.")

# Part 5: Business Insights
st.header("Business Insights")
st.write("""
The analysis shows a statistically significant difference in average view counts between verified and unverified accounts. This suggests that verified accounts may drive more viewership on average, potentially due to higher content quality or greater user trust in verified accounts.

Next steps could involve building a regression model to further explore the factors influencing view counts, such as content type, follower count, and engagement metrics.
""")

# Optional: Display raw data for inspection
st.sidebar.header("Analysis Options")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)
