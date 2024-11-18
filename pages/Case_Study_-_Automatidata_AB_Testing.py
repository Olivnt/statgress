import streamlit as st
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Title Section
st.title("Automatidata Project ")
st.markdown("### Case Study: Automatidata – Estimating Taxi Fares for New York City TLC")
st.write("""
This project analyzes taxi fare amounts in relation to payment types using statistical methods.
Insights from this study informed the development of a fare estimation app for the New York City Taxi and Limousine Commission (TLC).
""")

# Key Contributions Section
st.header("Key Contributions")
st.markdown("""
- **Descriptive Statistics:** Calculated key metrics like mean, median, and standard deviation for taxi fares.
- **Hypothesis Testing:** Conducted statistical hypothesis tests to identify significant differences in fare amounts by payment type.
- **Visualization and Insights:** Created compelling data visualizations to illustrate trends and findings.
- **Executive Summary:** Delivered a concise report summarizing actionable insights for stakeholders.
""")

# Part 1: Data Loading and Preparation
st.header("Part 1: Data Loading and Preparation")
@st.cache_data
def load_data():
    return pd.read_csv("myPortfolio/dataSets/2017_Yellow_Taxi_Trip_Data.csv", index_col=0)

taxi_data = load_data()
st.write("### Sample Data:")
st.dataframe(taxi_data.head())

# Descriptive Metrics Section
st.header("Part 2: Descriptive Metrics")
st.subheader("Key Statistics")
st.write("""
Below are some key descriptive statistics derived from the dataset, providing a quick overview of the key trends and characteristics:
""")
st.write(f"- **Total Rides:** {taxi_data.shape[0]}")
st.write(f"- **Average Fare Amount:** ${taxi_data['fare_amount'].mean():.2f}")
st.write(f"- **Median Fare Amount:** ${taxi_data['fare_amount'].median():.2f}")
st.write(f"- **Standard Deviation of Fare Amount:** ${taxi_data['fare_amount'].std():.2f}")

st.subheader("Fare Amount Distribution by Payment Type")
payment_type_mapping = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute", 5: "Unknown"}
taxi_data['payment_type'] = taxi_data['payment_type'].map(payment_type_mapping)
average_fares = taxi_data.groupby('payment_type')['fare_amount'].mean()

# Visualization
fig, ax = plt.subplots()
average_fares.plot(kind='bar', ax=ax)
ax.set_title("Average Fare Amount by Payment Type")
ax.set_ylabel("Average Fare ($)")
st.pyplot(fig)

# Hypothesis Testing Section
st.header("Part 3: Hypothesis Testing (A/B Test)")
st.markdown("""
**Null Hypothesis (H₀):** No difference in average fare amounts between credit card and cash payments.  
**Alternative Hypothesis (H₁):** A difference exists in average fare amounts between credit card and cash payments.
""")

# Perform the t-test
credit_card_fares = taxi_data[taxi_data['payment_type'] == "Credit Card"]['fare_amount']
cash_fares = taxi_data[taxi_data['payment_type'] == "Cash"]['fare_amount']
t_test_result = stats.ttest_ind(a=credit_card_fares, b=cash_fares, equal_var=False)

st.subheader("T-test Results")
st.write(f"**Statistic:** {t_test_result.statistic:.2f}")
st.write(f"**P-value:** {t_test_result.pvalue:.4f}")

alpha = 0.05
if t_test_result.pvalue < alpha:
    st.success("**Conclusion:** Reject the null hypothesis. Significant difference in fare amounts between payment types.")
else:
    st.warning("**Conclusion:** Fail to reject the null hypothesis. No significant difference detected.")

# Business Insights Section
st.header("Part 4: Business Insights")
st.write("""
This analysis highlights opportunities to refine pricing strategies by leveraging payment type insights. The data suggests that credit card payments are associated with higher fares, offering a potential avenue to maximize revenue. However, to establish causality and ensure robust decision-making, further investigations using controlled experimental designs are recommended.
""")

# Optional Sidebar with Raw Data Toggle
st.sidebar.header("Options")
if st.sidebar.checkbox("Show Raw Data (First 100 Rows)"):
    st.subheader("Raw Data Sample (First 100 Rows)")
    st.dataframe(taxi_data.head(100))
