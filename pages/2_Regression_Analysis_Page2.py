# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Title and Introduction
st.title("I use Multiple Linear Regression here")
st.write("""
Here I explore multiple linear regression, examining the relationship between promotional budgets (TV, Radio, Social Media) and sales.
I will load data, explore relationships, fit a regression model, and analyze model assumptions.
""")

# Step 1: Load Dataset
st.header("Step 1: Load Dataset")
data = pd.read_csv('myPortfolio/dataSets/marketing_sales_data.csv')  # Replace with your file path
st.write("Data preview:")
st.write(data.head())

# Step 2: Data Exploration
st.header("Step 2: Data Exploration")

# Pairplot of features
st.subheader("Pairplot of Features")
fig = sns.pairplot(data)
st.pyplot(fig)

# Mean sales per TV and Influencer categories
st.subheader("Mean Sales by TV and Influencer Categories")
mean_sales_tv = data.groupby('TV')['Sales'].mean()
mean_sales_influencer = data.groupby('Influencer')['Sales'].mean()
st.write("Mean Sales by TV Category:")
st.write(mean_sales_tv)
st.write("Mean Sales by Influencer Category:")
st.write(mean_sales_influencer)

# Drop rows with missing data
data = data.dropna()
data = data.rename(columns={'Social Media': 'Social_Media'})

# Step 3: Model Building
st.header("Step 3: Model Building")
st.write("Creating OLS model with Sales as the dependent variable and TV, Radio, and Social Media as predictors.")

# Define and fit the OLS model
ols_formula = 'Sales ~ C(TV) + Radio'
model = ols(formula=ols_formula, data=data).fit()
model_results = model.summary()
st.write("Model Summary:")
st.text(model_results)

# Scatterplots for independent vs dependent variables
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

# Residual analysis
st.subheader("Residual Analysis")
residuals = model.resid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Histogram of Residuals")
sm.qqplot(residuals, line='s', ax=ax2)
ax2.set_title("Normal QQ Plot")
st.pyplot(fig)

# Homoscedasticity Check
st.subheader("Homoscedasticity Check")
fig, ax = plt.subplots()
sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
ax.set_title("Fitted Values vs. Residuals")
st.pyplot(fig)

# Variance Inflation Factor (VIF) Calculation
st.header("Variance Inflation Factor (VIF)")
X = data[['Radio', 'Social_Media']]
vif_data = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
st.write("VIF Results:")
st.write(vif_data)
