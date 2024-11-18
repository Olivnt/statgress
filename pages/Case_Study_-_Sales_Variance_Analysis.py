# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Full Dataset"])

# Data Loading Function
@st.cache_data
def load_data():
    return pd.read_csv('dataSets\\marketing_sales_data.csv')

data = load_data()

# Main Page
if page == "Main Page":
    # Title and Introduction
    st.title("ANOVA Analysis: Comparing Sales Across TV Promotion Levels")
    st.markdown("### Case Study: Hypothesis Testing with ANOVA")
    st.write("""
    **Overview:**  
    This project investigates the impact of different TV promotion levels on sales using a one-way ANOVA test. The analysis helps identify significant differences in sales across categories and provides actionable insights for marketing strategies.
    """)

    # Key Contributions Section
    st.header("Key Contributions")
    st.markdown("""
    - **Data Visualization:** Explored sales distribution across TV promotion levels using boxplots.  
    - **Hypothesis Testing:** Conducted a one-way ANOVA test to evaluate differences in sales among groups.  
    - **Model Diagnostics:** Checked assumptions of normality and homoscedasticity for reliable results.  
    - **Post Hoc Testing:** Performed Tukey's HSD test to identify pairwise differences between groups.  
    """)

    # Step 1: Load Dataset
    st.header("Step 1: Load Dataset")
    st.write("### Sample Data:")
    st.dataframe(data.head())

    # Step 2: Data Visualization
    st.header("Step 2: Data Visualization")
    st.subheader("Boxplot of Sales by TV Promotion Level")
    fig, ax = plt.subplots()
    sns.boxplot(x="TV", y="Sales", data=data, ax=ax)
    ax.set_title("Sales by TV Promotion Level")
    st.pyplot(fig)

    st.subheader("Boxplot of Sales by Influencer")
    fig, ax = plt.subplots()
    sns.boxplot(x="Influencer", y="Sales", data=data, ax=ax)
    ax.set_title("Sales by Influencer Level")
    st.pyplot(fig)

    # Step 3: Data Preparation
    st.header("Step 3: Data Preparation")
    data_cleaned = data.dropna(axis=0).reset_index(drop=True)
    st.write("### Data after Removing Missing Values:")
    st.dataframe(data_cleaned.isnull().sum(axis=0))

    # Step 4: ANOVA Test
    st.header("Step 4: ANOVA Test")
    st.write("Performing ANOVA to compare sales across TV promotion levels.")
    model = ols('Sales ~ C(TV)', data=data_cleaned).fit()
    model_results = model.summary()
    st.write("Model Summary:")
    st.text(model_results)

    # Residual Analysis
    st.subheader("Residual Analysis")
    residuals = model.resid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title("Histogram of Residuals")
    sm.qqplot(residuals, line='s', ax=ax2)
    ax2.set_title("Normal Q-Q Plot")
    st.pyplot(fig)
    st.write("""
    **Residual Analysis Insight:**  
    - Residuals should follow a normal distribution for ANOVA results to be valid.  
    - The Q-Q plot and histogram help assess this assumption.
    """)

    # Homoscedasticity Check
    st.subheader("Homoscedasticity Check")
    fig, ax = plt.subplots()
    sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Fitted Values vs. Residuals")
    st.pyplot(fig)
    st.write("""
    **Homoscedasticity Insight:**  
    - Residuals should have constant variance across fitted values.  
    - Patterns or non-uniform spread indicate a violation of this assumption.
    """)

    # ANOVA Table
    st.subheader("ANOVA Table")
    anova_results = sm.stats.anova_lm(model, typ=2)
    st.write("### One-Way ANOVA Results:")
    st.dataframe(anova_results)

    # Step 5: Tukey's HSD Post Hoc Test
    st.header("Step 5: Tukey's HSD Post Hoc Test")
    st.write("""
    After finding a significant difference with ANOVA, Tukey's HSD test is used to identify which groups differ from each other.
    """)
    tukey_test = pairwise_tukeyhsd(endog=data_cleaned["Sales"], groups=data_cleaned["TV"])
    st.text(tukey_test.summary())

    # Business Insights Section
    st.header("Business Insights")
    st.write("""
    **Outcome:**  
    The ANOVA results highlight significant differences in sales across TV promotion levels. Tukey's HSD test further identifies specific group differences.  
    **Recommendations:**  
    - Focus marketing efforts on the TV promotion levels associated with the highest sales.  
    - Validate findings with additional predictors (e.g., seasonal trends or geographical data).  
    """)

# View Full Dataset Page
elif page == "View Full Dataset":
    st.title("View Full Dataset")
    st.write("""
    This page provides the complete dataset used in the analysis.  
    You can inspect the data structure and all values for additional insights.
    """)
    st.dataframe(data)
