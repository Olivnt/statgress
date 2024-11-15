# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Title and Introduction
st.title("Hypothesis Testing using ANOVA")
st.write("""
Here I explore hypothesis testing using ANOVA to compare sales across different levels of TV promotions.
It includes steps for data loading, visualization, model fitting, and post hoc testing.
""")

# Step 1: Load Dataset
st.header("Step 1: Load Dataset")
data = pd.read_csv('myPortfolio/dataSets/marketing_sales_data.csv')  # Replace with your file path
st.write("Data preview:")
st.write(data.head())

# Step 2: Data Visualization
st.header("Step 2: Data Visualization")

# Boxplot of Sales by TV
st.subheader("Boxplot of Sales by TV")
fig, ax = plt.subplots()
sns.boxplot(x="TV", y="Sales", data=data, ax=ax)
st.pyplot(fig)

# Boxplot of Sales by Influencer
st.subheader("Boxplot of Sales by Influencer")
fig, ax = plt.subplots()
sns.boxplot(x="Influencer", y="Sales", data=data, ax=ax)
st.pyplot(fig)

# Step 3: Data Preparation
st.header("Step 3: Data Preparation")
data = data.dropna(axis=0).reset_index(drop=True)
st.write("Data after removing missing values:")
st.write(data.isnull().sum(axis=0))

# Step 4: ANOVA Test
st.header("Step 4: ANOVA Test")

# Define and fit the OLS model
ols_formula = 'Sales ~ C(TV)'
model = ols(formula=ols_formula, data=data).fit()
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

# ANOVA Table
st.subheader("ANOVA Table")
anova_results = sm.stats.anova_lm(model, typ=2)
st.write("One-Way ANOVA Table:")
st.write(anova_results)

# Step 5: Post Hoc Test
st.header("Step 5: Tukey's HSD Post Hoc Test")
tukey_test = pairwise_tukeyhsd(endog=data["Sales"], groups=data["TV"])
st.write("Tukey HSD Post Hoc Test Results:")
st.text(tukey_test.summary())
