import streamlit as st
import pandas as pd
from scipy import stats

st.title("Automatidata Project: A/B Testing on NYC Taxi Fare Amounts by Payment Type")

# Part 1: Imports and Data Loading
st.header("Data Loading and Preparation")

# Load the dataset
@st.cache_data  # Cache the data to enhance performance
def load_data():
    return pd.read_csv("myPortfolio/dataSets/2017_Yellow_Taxi_Trip_Data.csv", index_col=0)

# Load and display data
taxi_data = load_data()
st.write("Loaded Dataset:", taxi_data.head())

# Part 2: Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis")

# Display descriptive statistics for the dataset
st.subheader("Descriptive Statistics")
st.write(taxi_data.describe(include='all'))

# Calculate and display the average fare amount for each payment type
st.subheader("Average Fare Amount by Payment Type")
# Mapping for clarity
payment_type_mapping = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute", 5: "Unknown"}
taxi_data['payment_type'] = taxi_data['payment_type'].map(payment_type_mapping)
average_fares = taxi_data.groupby('payment_type')['fare_amount'].mean()
st.write(average_fares)

# Focus on Credit Card and Cash only for A/B testing
credit_card_fares = taxi_data[taxi_data['payment_type'] == "Credit Card"]['fare_amount']
cash_fares = taxi_data[taxi_data['payment_type'] == "Cash"]['fare_amount']

# Part 3: Hypothesis Testing (A/B Test)
st.header("Hypothesis Testing: A/B Test on Fare Amount by Payment Type")

# Define the hypotheses
st.write("**Hypotheses**")
st.write("**Null Hypothesis (H₀):** There is no difference in the average fare amount between customers who use credit cards and those who use cash.")
st.write("**Alternative Hypothesis (H₁):** There is a difference in the average fare amount between customers who use credit cards and those who use cash.")

# Perform a two-sample t-test (Welch's t-test) between Credit Card and Cash fares
t_test_result = stats.ttest_ind(a=credit_card_fares, b=cash_fares, equal_var=False)

# Display the t-test results
st.subheader("T-test Results")
st.write(f"Statistic: {t_test_result.statistic}")
st.write(f"P-value: {t_test_result.pvalue}")

# Interpret the p-value
alpha = 0.05  # Significance level of 5%
if t_test_result.pvalue < alpha:
    st.write("**Conclusion:** Reject the null hypothesis. There is a statistically significant difference in the average fare amount between credit card and cash payments.")
else:
    st.write("**Conclusion:** Fail to reject the null hypothesis. There is no statistically significant difference in the average fare amount between credit card and cash payments.")

# Part 4: Business Insights
st.header("Business Insights")
st.write("""
The key insight from this A/B test is that encouraging customers to pay with credit cards may generate more revenue for taxi drivers if there is a statistically significant difference in fare amounts. However, it is important to note the assumptions of this test: the dataset assumes an experimental grouping, which is not how the data was actually collected. 
Further analysis or a true experimental setup would be required to draw causal conclusions.
""")

# Optional: Display raw data for inspection
st.sidebar.header("Analysis Options")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(taxi_data)
