import streamlit as st
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Full Dataset"])

# Data Loading Function
@st.cache_data  # Cache data to optimize performance
def load_data():
    return pd.read_csv('myPortfolio/dataSets/waze_dataset.csv')

df = load_data()

# Main Page
if page == "Main Page":
    st.markdown("### Case Study: Waze – Predicting User Churn with Statistical Insights")

    # Overview Section
    st.write("""
    **Overview:**
    This project predicts user churn by analyzing monthly user behavior. The analysis investigates the relationship between ride frequency and device type using statistical hypothesis testing. The insights demonstrate practical applications of data analytics in addressing user retention challenges.
    """)

    # Key Contributions Section
    st.header("Key Contributions")
    st.markdown("""
    - **Descriptive Analysis:** Computed statistical measures (mean, variance, etc.) to explore user ride patterns across device types.
    - **Two-Sample Hypothesis Test:** Applied Python-based hypothesis testing to assess if device type significantly influenced average monthly rides.
    - **Data Visualizations:** Created engaging visuals (e.g., boxplots and bar graphs) to illustrate user behavior differences.
    - **Insights and Recommendations:** Highlighted strategies based on the analysis that could guide improvements in user retention.
    """)

    # Data Loading and Preparation Section
    st.header("Part 1: Data Loading and Preparation")
    st.write("### Sample Data:")
    st.dataframe(df.head())

    # Descriptive Metrics Section
    st.header("Part 2: Descriptive Metrics")
    st.subheader("Key Statistics")
    st.write("""
    Below are key descriptive statistics, providing a snapshot of user behavior across device types:
    """)
    st.write(f"- **Total Users:** {df.shape[0]}")
    st.write(f"- **Average Monthly Drives:** {df['drives'].mean():.2f}")
    st.write(f"- **Median Monthly Drives:** {df['drives'].median():.2f}")
    st.write(f"- **Standard Deviation of Monthly Drives:** {df['drives'].std():.2f}")

    st.subheader("Average Drives by Device Type")
    # Map device types for clarity
    map_dictionary = {'Android': 2, 'iPhone': 1}
    df['device_type'] = df['device'].map(map_dictionary)
    average_drives = df.groupby('device')['drives'].mean()

    # Visualization
    fig, ax = plt.subplots()
    average_drives.plot(kind='bar', ax=ax, color=['blue', 'orange'])
    ax.set_title("Average Drives by Device Type")
    ax.set_ylabel("Average Drives")
    st.pyplot(fig)

    # Hypothesis Testing Section
    st.header("Part 3: Hypothesis Testing")
    st.markdown("""
    **Hypothesis Testing**  
    - **Null Hypothesis (H₀):** There is no difference in the average number of drives between iPhone and Android users.  
    - **Alternative Hypothesis (H₁):** There is a difference in the average number of drives between iPhone and Android users.
    """)

    # Isolate the `drives` data for each device type
    iPhone_drives = df[df['device_type'] == 1]['drives']
    Android_drives = df[df['device_type'] == 2]['drives']

    # Perform a two-sample t-test with unequal variances
    t_test_result = stats.ttest_ind(a=iPhone_drives, b=Android_drives, equal_var=False)

    # Display the t-test results
    st.subheader("T-test Results")
    st.write(f"**Statistic:** {t_test_result.statistic:.2f}")
    st.write(f"**P-value:** {t_test_result.pvalue:.4f}")

    # Interpret the results
    alpha = 0.05  # Significance level of 5%
    if t_test_result.pvalue < alpha:
        st.success("""
        **Conclusion:** Reject the null hypothesis.  
        There is a statistically significant difference in the average number of drives between iPhone and Android users.
        This conclusion is based on the p-value being less than the significance level (α = 0.05).
        """)
    else:
        st.warning("""
        **Conclusion:** Fail to reject the null hypothesis.  
        There is no statistically significant difference in the average number of drives between iPhone and Android users.
        This conclusion is based on the p-value being greater than the significance level (α = 0.05), meaning we do not have enough evidence to suggest a difference.
        """)

    # Business Insights Section
    st.header("Part 4: Business Insights")
    st.write("""
    **Outcome:**
    This analysis highlights the potential for device-specific strategies to improve user retention. The dataset provides an opportunity to practice real-world data analytics techniques and explore how statistical insights can address user churn challenges.

    **Portfolio Highlights:**
    - **Technical Expertise:** Demonstrated proficiency in Python, hypothesis testing, and data visualization tools such as Matplotlib and Seaborn.
    - **Statistical Skills:** Applied descriptive and inferential statistics to extract actionable insights from data.
    - **Analytical Communication:** Delivered insights in a professional and accessible format for diverse audiences.
    - **Problem-Solving:** Explored data-driven strategies to address user behavior challenges in a hypothetical scenario.
    """)

# Full Dataset Page
elif page == "View Full Dataset":
    st.title("View Full Dataset")
    st.markdown("""
    This page provides the complete dataset used in the analysis.  
    You can inspect the data structure and all values for additional insights.
    """)
    st.dataframe(df)
