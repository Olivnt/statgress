# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Title and introduction
st.title("Using Simple Linear Regression")
st.write("""
Here I explore the relationship between promotional budgets (TV, Radio, Social Media) and sales using simple linear regression. 
I will load data, explore relationships, fit a regression model, and analyze model assumptions.
""")

# Step 1: Load Dataset
st.header("Step 1: Load Dataset")

# Load the dataset directly from the file path
data = pd.read_csv('myPortfolio/dataSets/marketing_and_sales_data_evaluate_lr.csv')
st.write("Data preview:")
st.write(data.head())

# Step 2: Data Exploration
st.header("Step 2: Data Exploration")

# Display dataset shape
st.write(f"Dataset has {data.shape[0]} rows and {data.shape[1]} columns.")

# Show descriptive statistics for promotional budgets
st.subheader("Descriptive Statistics for Promotional Budgets")
st.write(data[['TV', 'Radio', 'Social_Media']].describe())

# Show percentage of missing values in Sales
missing_sales = round(data['Sales'].isna().mean() * 100, 2)
st.write(f"Percentage of missing values in Sales: {missing_sales}%")

# Drop rows with missing Sales values
data = data.dropna(subset=['Sales'])

# Histogram of Sales
st.subheader("Sales Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Sales'], kde=True, ax=ax)
ax.set_title("Distribution of Sales")
st.pyplot(fig)  # Pass the fig object here

# Step 3: Model Building
st.header("Step 3: Model Building")

# Pairplot for visual inspection
st.subheader("Pairplot of Features")
fig = sns.pairplot(data)  # Pairplot directly creates a figure
st.pyplot(fig)  # Pass the fig object here

### YOUR CODE HERE ### 

# Define the OLS formula.
ols_formula = 'Sales ~ TV'

# Create an OLS model.
OLS = ols(formula = ols_formula, data = data)

# Fit the model.
model = OLS.fit()

# Save the results summary.
model_results = model.summary()

# Display the model results.
st.write(model_results)

# Step 4: Model Assumption Checks
st.header("Step 4: Model Assumption Checks")

# Linearity
st.subheader("Linearity Check")
fig, ax = plt.subplots()
sns.scatterplot(x=data['TV'], y=data['Sales'], ax=ax)
ax.set_title("TV Promotion Budget vs. Sales")
st.pyplot(fig)  # Pass the fig object here

# Normality
st.subheader("Normality Check")
residuals = model.resid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(residuals, kde=True, ax=ax1)
sm.qqplot(residuals, line='s', ax=ax2)
ax1.set_title("Residuals Distribution")
ax2.set_title("Normal Q-Q plot")
st.pyplot(fig)  # Pass the fig object here

# Homoscedasticity
st.subheader("Homoscedasticity Check")
fig, ax = plt.subplots()
sns.scatterplot(x=model.fittedvalues, y=residuals, ax=ax)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
ax.set_title("Fitted Values vs. Residuals")
st.pyplot(fig)  # Pass the fig object here


# Step 5: Results and Recommendations
st.header("Step 5: Results and Recommendations")

# Display R-squared
r_squared = model.rsquared
st.write(f"R-squared: {r_squared:.3f}")

# Display coefficients
st.write("Model Coefficients:")
st.write(f"Intercept: {model.params['Intercept']:.4f}")
st.write(f"TV Coefficient: {model.params['TV']:.4f}")

# Interpretation
st.subheader("Interpretation and Recommendations")
st.write("""
Based on the model, an increase of $1 million in the TV promotional budget is associated with an increase of approximately 3.5614 million in sales.
TV promotion has the strongest impact on sales, so the company should prioritize this marketing channel.
""")
