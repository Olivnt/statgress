import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Full Dataset"])

# Data Loading Function
@st.cache_data
def load_data():
    return pd.read_csv("myPortfolio/dataSets/marketing_and_sales_data_evaluate_lr.csv")

data = load_data()

# Main Page
if page == "Main Page":
    # Title and introduction
    st.title("Marketing and Sales Analysis: Exploring the Impact of Promotional Budgets on Sales")
    st.markdown("### Case Study: Regression Analysis of Marketing Spend")
    st.write("""
    **Overview:**  
    This project explores the relationship between promotional budgets (TV, Radio, and Social Media) and sales using simple linear regression. The analysis identifies the impact of marketing spend on sales performance and provides actionable recommendations for resource allocation.
    """)

    # Key Contributions Section
    st.header("Key Contributions")
    st.markdown("""
    - **Exploratory Data Analysis:** Conducted descriptive statistics and visualized data distributions to understand sales trends.
    - **Regression Modeling:** Built and evaluated a simple linear regression model to analyze the relationship between TV promotional budgets and sales.
    - **Model Diagnostics:** Checked model assumptions, including linearity, normality, and homoscedasticity, to ensure robust results.
    - **Business Insights:** Delivered actionable insights to optimize marketing spend allocation.
    """)

    # Step 1: Load Dataset
    st.header("Step 1: Load Dataset")
    st.write("### Sample Data:")
    st.dataframe(data.head())

    # Step 2: Data Exploration
    st.header("Step 2: Data Exploration")
    st.write(f"Dataset has {data.shape[0]} rows and {data.shape[1]} columns.")

    st.subheader("Descriptive Statistics for Promotional Budgets")
    st.write(data[['TV', 'Radio', 'Social_Media']].describe())

    missing_sales = round(data['Sales'].isna().mean() * 100, 2)
    st.write(f"Percentage of missing values in Sales: {missing_sales}%")

    # Drop rows with missing Sales values
    data = data.dropna(subset=['Sales'])

    # Histogram of Sales
    st.subheader("Sales Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['Sales'], kde=True, ax=ax)
    ax.set_title("Distribution of Sales")
    st.pyplot(fig)

    # Step 3: Model Building
    st.header("Step 3: Model Building")
    st.subheader("Pairplot of Features")
    fig = sns.pairplot(data)
    st.pyplot(fig)

    ols_formula = 'Sales ~ TV'
    model = ols(formula=ols_formula, data=data).fit()
    model_results = model.summary()
    st.write(model_results)

    # Step 4: Model Assumption Checks
    st.header("Step 4: Model Assumption Checks")

    # Linearity
    st.subheader("Linearity Check")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['TV'], y=data['Sales'], ax=ax)
    ax.set_title("TV Promotion Budget vs. Sales")
    st.pyplot(fig)

    # Normality
    st.subheader("Normality Check")
    residuals = model.resid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(residuals, kde=True, ax=ax1)
    sm.qqplot(residuals, line='s', ax=ax2)
    ax1.set_title("Residuals Distribution")
    ax2.set_title("Normal Q-Q plot")
    st.pyplot(fig)

    # Homoscedasticity
    st.subheader("Homoscedasticity Check")
    fig, ax = plt.subplots()
    sns.scatterplot(x=model.fittedvalues, y=residuals, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Fitted Values vs. Residuals")
    st.pyplot(fig)

    # Step 5: Results and Recommendations
    st.header("Step 5: Results and Recommendations")
    st.write(f"**R-squared:** {model.rsquared:.3f}")
    st.write(f"""
    **R-squared Insight:**  
    The R-squared value of **{model.rsquared:.3f}** indicates the proportion of variability in sales that can be explained by the TV promotional budget. 
    This suggests that the model is effective in capturing the relationship, adding confidence to the reliability of predictions.
    """)

    st.write("Model Coefficients:")
    st.write(f"Intercept: {model.params['Intercept']:.4f}")
    st.write(f"TV Coefficient: {model.params['TV']:.4f}")

    st.subheader("Interpretation and Recommendations")
    st.write("""
    - **Interpretation:** An increase of $1 million in the TV promotional budget is associated with an increase of approximately 3.5614 million in sales.  
    - **Recommendation:** TV promotion has the strongest impact on sales, and the company should prioritize this channel.
    """)

# View Full Dataset Page
elif page == "View Full Dataset":
    st.title("View Full Dataset")
    st.write("""
    This page provides the complete dataset used in the analysis.  
    You can inspect the data structure and all values for additional insights.
    """)
    st.dataframe(data)
