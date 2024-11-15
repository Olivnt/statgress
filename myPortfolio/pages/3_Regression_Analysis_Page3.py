# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Introduction
st.title("Performing Logistic Regression")
st.write("""
Here I explore logistic regression to predict customer satisfaction based on inflight entertainment.
It includes steps for data loading, preprocessing, model fitting, and performance evaluation.
""")

# Step 1: Load Dataset
st.header("Step 1: Load Dataset")
df_original = pd.read_csv('myPortfolio/dataSets/Invistico_Airline.csv')  # Replace with your file path
st.write("Data preview:")
st.write(df_original.head(10))

# Step 2: Data Exploration
st.header("Step 2: Data Exploration")

# Display data types
st.subheader("Data Types")
st.write(df_original.dtypes)

# Count of satisfaction values
st.subheader("Satisfaction Value Counts")
st.write(df_original['satisfaction'].value_counts(dropna=False))

# Check for missing values
st.subheader("Missing Values")
st.write(df_original.isnull().sum())

# Drop rows with missing data
df_subset = df_original.dropna(axis=0).reset_index(drop=True)

# Convert "Inflight entertainment" to float
df_subset = df_subset.astype({"Inflight entertainment": float})

# Encode "satisfaction" column
encoder = OneHotEncoder(drop='first')
df_subset['satisfaction'] = encoder.fit_transform(df_subset[['satisfaction']]).toarray()
st.write("Data after preprocessing:")
st.write(df_subset.head(10))

# Step 3: Logistic Regression Model
st.header("Step 3: Logistic Regression Model")

# Define X and y, and split data
X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit logistic regression model
clf = LogisticRegression().fit(X_train, y_train)
st.write("Model Coefficients:")
st.write(f"Coefficient: {clf.coef_[0][0]:.6f}")
st.write(f"Intercept: {clf.intercept_[0]:.6f}")

# Step 4: Model Visualization
st.header("Step 4: Model Visualization")

# Plot logistic regression line
fig, ax = plt.subplots()
sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None, ax=ax)
ax.set_title("Logistic Regression Line")
st.pyplot(fig)

# Step 5: Predictions and Evaluation
st.header("Step 5: Predictions and Evaluation")

# Predictions on test set
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Display predictions
st.subheader("Predictions")
st.write("Predicted values (0 or 1):")
st.write(y_pred)
st.write("Predicted probabilities:")
st.write(y_pred_proba)

# Evaluation Metrics
st.subheader("Evaluation Metrics")
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.6f}")
st.write(f"Precision: {precision:.6f}")
st.write(f"Recall: {recall:.6f}")
st.write(f"F1 Score: {f1_score:.6f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
fig, ax = plt.subplots()
disp.plot(ax=ax)
st.pyplot(fig)
