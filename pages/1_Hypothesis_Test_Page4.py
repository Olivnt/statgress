import streamlit as st
import pandas as pd
from scipy import stats

# Streamlit app title
st.title("Waze Project: Statistical Analysis of User Rides by Device Type")

# Part 1: Imports and Data Loading
st.header("Data Loading and Preparation")

# Load the dataset
@st.cache_data  # Cache data to optimize performance
def load_data():
    return pd.read_csv('myPortfolio/dataSets/waze_dataset.csv')

# Load and display data
df = load_data()
st.write("Loaded Dataset:", df.head())

# Map device types to numerical values for analysis
map_dictionary = {'Android': 2, 'iPhone': 1}
df['device_type'] = df['device'].map(map_dictionary)

# Part 2: Data Exploration and Descriptive Statistics
st.header("Descriptive Statistics")

# Calculate and display average number of drives for each device type
average_drives = df.groupby('device_type')['drives'].mean()
st.write("Average drives by device type:\n", average_drives)

# Part 3: Hypothesis Testing
st.header("Hypothesis Testing")

# Isolate the `drives` data for each device type
iPhone_drives = df[df['device_type'] == 1]['drives']
Android_drives = df[df['device_type'] == 2]['drives']

# Conduct a two-sample t-test with unequal variances
t_test_result = stats.ttest_ind(a=iPhone_drives, b=Android_drives, equal_var=False)

# Display the t-test results
st.write("T-test Result:")
st.write(f"Statistic: {t_test_result.statistic}")
st.write(f"P-value: {t_test_result.pvalue}")

# Interpret the p-value
alpha = 0.05  # Significance level of 5%
if t_test_result.pvalue < alpha:
    st.write("**Result:** Reject the null hypothesis. There is a statistically significant difference in the average number of drives between iPhone and Android users.")
else:
    st.write("**Result:** Fail to reject the null hypothesis. There is no statistically significant difference in the average number of drives between iPhone and Android users.")

# Optional: Display interactive elements for further analysis
st.sidebar.header("Analysis Options")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df)
