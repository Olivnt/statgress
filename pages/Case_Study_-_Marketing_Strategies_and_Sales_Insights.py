# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Full Dataset"])

# Data Loading Function
@st.cache_data
def load_data():
    return pd.read_csv('myPortfolio/dataSets/marketing_sales_data.csv')

data = load_data()

# Main Page
if page == "Main Page":
    # Title and Introduction
    st.title("Multiple Linear Regression Analysis: Impact of Marketing Strategies on Sales")
    st.markdown("### Case Study: Analyzing Marketing Spend to Optimize Sales")
    st.write("""
    **Overview:**  
    This project explores the relationship between promotional budgets (TV, Radio, and Social Media) and sales using multiple linear regression. 
    The analysis identifies how different marketing strategies influence sales performance, providing actionable insights for resource allocation.
    """)

    # Key Contributions Section
    st.header("Key Contributions")
    st.markdown("""
    - **Exploratory Data Analysis:** Visualized relationships between marketing spend and sales, and examined group-wise mean sales.  
    - **Regression Modeling:** Built and evaluated a multiple linear regression model to quantify the impact of various predictors on sales.  
    - **Model Diagnostics:** Conducted assumption checks, including residual analysis, normality, and multicollinearity (VIF).  
    - **Business Insights:** Delivered actionable recommendations to optimize marketing strategies and maximize sales.  
    """)

    # Step 1: Load Dataset
    st.header("Step 1: Load Dataset")
    st.write("### Sample Data:")
    st.dataframe(data.head())

    # Step 2: Data Exploration
    st.header("Step 2: Data Exploration")
    st.subheader("Pairplot of Features")
    fig = sns.pairplot(data)
    st.pyplot(fig)

    st.subheader("Mean Sales by TV and Influencer Categories")
    mean_sales_tv = data.groupby('TV')['Sales'].mean()
    mean_sales_influencer = data.groupby('Influencer')['Sales'].mean()
    st.write("Mean Sales by TV Category:")
    st.write(mean_sales_tv)
    st.write("Mean Sales by Influencer Category:")
    st.write(mean_sales_influencer)

    # Drop rows with missing data
    data = data.dropna().rename(columns={'Social Media': 'Social_Media'})

    # Step 3: Model Building
    st.header("Step 3: Model Building")
    st.write("Creating OLS model with Sales as the dependent variable and TV, Radio, and Social Media as predictors.")
    ols_formula = 'Sales ~ C(TV) + Radio'
    model = ols(formula=ols_formula, data=data).fit()
    model_results = model.summary()
    st.write("Model Summary:")
    st.text(model_results)

    st.subheader("Scatterplots of Features vs. Sales")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(x=data['Radio'], y=data['Sales'], ax=axes[0])
    axes[0].set_title("Radio and Sales")
    sns.scatterplot(x=data['Social_Media'], y=data['Sales'], ax=axes[1])
    axes[1].set_title("Social Media and Sales")
    axes[1].set_xlabel("Social Media")
    st.pyplot(fig)

    # Step 4: Model Assumption Checks
    st.header("Step 4: Model Assumption Checks")

    # Residual Analysis
    st.subheader("Residual Analysis")
    residuals = model.resid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title("Histogram of Residuals")
    sm.qqplot(residuals, line='s', ax=ax2)
    ax2.set_title("Normal QQ Plot")
    st.pyplot(fig)
    st.write("""
    **Residual Analysis Insight:**  
    - The histogram of residuals and the Q-Q plot help check the normality assumption.  
    - If residuals follow a normal distribution, the model's predictions are more reliable.
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
    - Residuals should have constant variance around the fitted values.  
    - Patterns or funnel shapes indicate a violation of this assumption.
    """)

    # Variance Inflation Factor (VIF)
    st.header("Variance Inflation Factor (VIF)")
    X = data[['Radio', 'Social_Media']]
    vif_data = pd.DataFrame({
        'Feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    st.write("VIF Results:")
    st.write(vif_data)
    st.write("""
    **Variance Inflation Factor Insight:**  
    - VIF measures multicollinearity (correlation among predictors).  
    - A VIF > 10 suggests high multicollinearity, which can distort the model's coefficient estimates.
    """)

    # Business Insights Section
    st.header("Business Insights")
    st.write("""
    **Outcome:**  
    The multiple linear regression model highlights the significant impact of TV and Radio promotional budgets on sales.  
    **Recommendations:**  
    - Allocate resources strategically based on the predictors’ influence on sales.  
    - Use variance inflation factor (VIF) to monitor and reduce multicollinearity for better model performance.  
    - Investigate additional predictors (e.g., seasonal trends) to enhance model accuracy.  
    """)

# View Full Dataset Page
elif page == "View Full Dataset":
    st.title("View Full Dataset")
    st.write("""
    This page provides the complete dataset used in the analysis.  
    You can inspect the data structure and all values for additional insights.
    """)
    st.dataframe(data)
