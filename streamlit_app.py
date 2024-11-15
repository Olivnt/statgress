import streamlit as st

# Set the title of the portfolio
st.title("Welcome to My Portfolio")

# Display an image with the updated parameter
st.image("myPortfolio/images/Profile-Foto.png", caption="Applied Statistics and Regression Analysis in Python", use_container_width=True)

# Add a brief introduction
st.write("""
Hello! I'm Olivier Ntahiraja, and this is my portfolio showcasing my work in data science, statistics, regression analysis, and programming.
Explore my projects, experience, and accomplishments.
""")

# Add contact information
st.write("### Contact Information")
st.write("""
- **Email**: [olivierntahiraja@gmail.com](mailto:olivierntahiraja@gmail.com)
- **Phone**: +47 458 43 753
""")
