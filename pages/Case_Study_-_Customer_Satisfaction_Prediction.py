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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Full Dataset"])

# Data Loading Function
@st.cache_data
def load_data():
    return pd.read_csv('myPortfolio/dataSets/Invistico_Airline.csv')

data = load_data()

# Main Page
if page == "Main Page":
    # Title and Introduction
    st.title("Logistic Regression Analysis: Predicting Customer Satisfaction")
    st.markdown("### Case Study: Logistic Regression for Airline Customer Satisfaction")
    st.write("""
    **Overview:**  
    This project explores the relationship between inflight entertainment ratings and customer satisfaction using logistic regression. The analysis aims to identify factors driving satisfaction and provides actionable insights for airlines to improve customer experiences.
    """)

    # Key Contributions Section
    st.header("Key Contributions")
    st.markdown("""
    - **Data Exploration:** Investigated data structure, handled missing values, and encoded categorical variables for modeling.  
    - **Logistic Regression Model:** Built and evaluated a logistic regression model to predict customer satisfaction.  
    - **Model Visualization:** Visualized the logistic regression line to demonstrate the relationship between predictors and outcomes.  
    - **Evaluation Metrics:** Assessed model performance using accuracy, precision, recall, and F1-score.  
    - **Insights:** Delivered recommendations to airlines for enhancing customer satisfaction strategies.  
    """)

    # Step 1: Load Dataset
    st.header("Step 1: Load Dataset")
    st.write("### Sample Data:")
    st.dataframe(data.head(10))

    # Step 2: Data Exploration
    st.header("Step 2: Data Exploration")
    st.subheader("Data Types")
    st.write(data.dtypes)

    st.subheader("Satisfaction Value Counts")
    st.write(data['satisfaction'].value_counts(dropna=False))

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Drop rows with missing data and preprocess
    data_cleaned = data.dropna(axis=0).reset_index(drop=True)
    data_cleaned = data_cleaned.astype({"Inflight entertainment": float})
    encoder = OneHotEncoder(drop='first')
    data_cleaned['satisfaction'] = encoder.fit_transform(data_cleaned[['satisfaction']]).toarray()
    st.write("Data after preprocessing:")
    st.dataframe(data_cleaned.head(10))

    # Step 3: Logistic Regression Model
    st.header("Step 3: Logistic Regression Model")
    X = data_cleaned[["Inflight entertainment"]]
    y = data_cleaned["satisfaction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression().fit(X_train, y_train)
    st.write("Model Coefficients:")
    st.write(f"Coefficient: {clf.coef_[0][0]:.6f}")
    st.write(f"Intercept: {clf.intercept_[0]:.6f}")

    # Step 4: Model Visualization
    st.header("Step 4: Model Visualization")
    fig, ax = plt.subplots()
    sns.regplot(x="Inflight entertainment", y="satisfaction", data=data_cleaned, logistic=True, ci=None, ax=ax)
    ax.set_title("Logistic Regression Line")
    st.pyplot(fig)

    # Step 5: Predictions and Evaluation
    st.header("Step 5: Predictions and Evaluation")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    st.subheader("Predictions")
    st.write("Predicted values (0 or 1):")
    st.write(y_pred)
    st.write("Predicted probabilities:")
    st.write(y_pred_proba)

    st.subheader("Evaluation Metrics")
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.6f}")
    st.write(f"Precision: {precision:.6f}")
    st.write(f"Recall: {recall:.6f}")
    st.write(f"F1 Score: {f1_score:.6f}")

    # Metrics Explanation
    st.subheader("Metrics Explanation")
    st.markdown("""
    - **Accuracy:** Measures the proportion of correctly classified instances. A high accuracy indicates the model performs well overall.
    - **Precision:** Indicates the proportion of correctly classified positive instances (satisfied customers) among all predicted positives. High precision reduces false positives.
    - **Recall:** Reflects the proportion of correctly identified satisfied customers among all actual satisfied customers. High recall minimizes false negatives.
    - **F1 Score:** Balances precision and recall, especially useful when the dataset is imbalanced. A high F1 score indicates strong model performance in both metrics.
    """)

    st.subheader("Confusion Matrix")
    cm = metrics.confusion_matrix(y_test, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Business Insights Section
    st.header("Business Insights")
    st.write("""
    **Outcome:**  
    The logistic regression model effectively predicts customer satisfaction based on inflight entertainment ratings.  
    **Recommendations:**  
    - Prioritize enhancing inflight entertainment quality to improve overall customer satisfaction.  
    - Further investigate additional factors influencing satisfaction for comprehensive insights.  
    """)

# View Full Dataset Page
elif page == "View Full Dataset":
    st.title("View Full Dataset")
    st.write("""
    This page provides the complete dataset used in the analysis.  
    You can inspect the data structure and all values for additional insights.
    """)
    st.dataframe(data)
