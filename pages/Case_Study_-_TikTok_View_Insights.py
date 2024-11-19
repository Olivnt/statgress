# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "View Full Dataset"])

# Data Loading Function
@st.cache_data
def load_data():
    return pd.read_csv("myPortfolio/dataSets/tiktok_dataset.csv")

data = load_data()

# Main Page
if page == "Main Page":
    # Title and Introduction
    st.title("Logistic Regression Analysis: Predicting Viewership on TikTok")
    st.markdown("### Case Study: Logistic Regression for TikTok Verified Accounts")
    st.write("""
    **Overview:**  
    This project analyzes viewership patterns on TikTok based on account verification status. The logistic regression model predicts whether verified accounts tend to generate higher viewership, helping stakeholders understand platform dynamics.
    """)

    # Key Contributions Section
    st.header("Key Contributions")
    st.markdown("""
    - **Data Exploration:** Investigated data structure, handled missing values, and encoded categorical variables for modeling.  
    - **Logistic Regression Model:** Built and evaluated a logistic regression model to assess viewership trends.  
    - **Model Visualization:** Visualized relationships between verification status and viewership.  
    - **Insights:** Delivered data-driven recommendations for improving platform engagement strategies.  
    """)

    # Step 1: Load Dataset
    st.header("Step 1: Load Dataset")
    st.write("### Sample Data:")
    st.dataframe(data.head(10))

    # Step 2: Data Cleaning
    st.header("Step 2: Data Cleaning")
    st.subheader("Handling Missing Values")
    missing_values = data.isna().sum()
    st.write("Missing Values in Each Column:")
    st.dataframe(missing_values)

    # Drop rows with missing values
    data_cleaned = data.dropna()
    st.write("### Data After Dropping Missing Values:")
    st.dataframe(data_cleaned.head(10))

    # Step 3: Logistic Regression Model
    st.header("Step 3: Logistic Regression Model")

    # Encode "verified_status" column
    encoder = OneHotEncoder(drop="first")
    data_cleaned["verified_status"] = encoder.fit_transform(data_cleaned[["verified_status"]]).toarray()

    # Define X and y, and split data
    X = data_cleaned[["video_view_count"]]
    y = data_cleaned["verified_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit logistic regression model
    clf = LogisticRegression().fit(X_train, y_train)
    st.write("Model Coefficients:")
    st.write(f"Coefficient: {clf.coef_[0][0]:.6f}")
    st.write(f"Intercept: {clf.intercept_[0]:.6f}")

    # Step 4: Model Visualization
    st.header("Step 4: Model Visualization")
    fig, ax = plt.subplots()
    sns.regplot(x="video_view_count", y="verified_status", data=data_cleaned, logistic=True, ci=None, ax=ax)
    ax.set_title("Logistic Regression Line: Viewership vs. Verification Status")
    st.pyplot(fig)

    # Step 5: Predictions and Evaluation
    st.header("Step 5: Predictions and Evaluation")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    st.subheader("Predictions")
    st.write("Predicted Verification Status (0 or 1):")
    st.write(y_pred)
    st.write("Predicted Probabilities:")
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
    - **Accuracy:** Measures the proportion of correctly classified instances. A high accuracy indicates good overall model performance.  
    - **Precision:** Proportion of correctly predicted positives out of all predicted positives. High precision minimizes false positives.  
    - **Recall:** Proportion of actual positives correctly identified by the model. High recall minimizes false negatives.  
    - **F1 Score:** Balances precision and recall, providing a single metric to evaluate the model's effectiveness, especially for imbalanced datasets.  
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
    Verified accounts are more likely to attract higher viewership, as demonstrated by the logistic regression model.  
    **Recommendations:**  
    - Prioritize verified accounts in promotional campaigns to maximize engagement.  
    - Investigate other factors (e.g., content type, follower count) to further understand viewership trends.  
    """)

# View Full Dataset Page
elif page == "View Full Dataset":
    st.title("View Full Dataset")
    st.write("""
    This page provides the complete dataset used in the analysis.  
    You can inspect the data structure and all values for additional insights.
    """)
    st.dataframe(data)
