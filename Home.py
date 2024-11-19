import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

# Home Page
if page == "Home":
    # Set the title of the portfolio
    st.title("Olivier Ntahiraja: Data Science Portfolio")

    # Display an image with the updated parameter
    st.image("myPortfolio/images/Profile-Foto.png", caption="Applied Statistics and Regression Analysis in Python", use_container_width=True)

    # Add a brief introduction
    st.write("""
    Welcome to the portfolio of Olivier Ntahiraja. This collection highlights my expertise in data science, statistical analysis, regression modeling, and programming.
Explore a curated selection of my projects, professional experience, and accomplishments that demonstrate my commitment to solving complex problems through data-driven insights.
    """)

    # Add contact information
    st.write("### Contact Information")
    st.write("""
    - **Email**: [olivierntahiraja@gmail.com](mailto:olivierntahiraja@gmail.com)
    - **Phone**: +47 458 43 753
    """)

# About Page
elif page == "About":
    st.title("About")
    st.write("""
    Welcome to the "About" section of my portfolio. Here, you will find important details regarding the datasets used in my projects.
    """)

    # Add dataset disclaimer
    st.write("""
    ### Data Sources
    - The datasets used in my projects come from publicly available sources, primarily Kaggle.
    - Some datasets may be fictive or simulated for educational purposes, as indicated in specific projects.
    - The analyses and insights provided are for demonstration purposes only and do not reflect real-world collaboration with the original data owners.
    """)

    # Add acknowledgment
    st.write("""
    ### Acknowledgments
    I acknowledge the efforts of dataset creators and platforms like Kaggle for providing valuable resources to the data science community.
    """)
