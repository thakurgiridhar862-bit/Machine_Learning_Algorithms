import streamlit as st
import pandas as pd
import numpy as np

# sklean ki libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# page title set krna
st.set_page_config(
    page_title="Placement Package Predictor", page_icon="🎓", layout="centered"
)

st.title("🎓 Placement Package Predictor")
st.write(
    "This project is built using Multiple Linear Regression to estimate placement packages."
)

# data read
df = pd.read_csv("data/placement_package_regression.csv")

with st.expander("📁 View Dataset Preview"):
    st.dataframe(df.head())

# data featuring
X = df[["CGPA", "Projects", "DSA_Score", "Internships", "Communication_Score"]]
Y = df["Package_LPA"]

# data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, Y_train)

st.sidebar.header("Enter Student Details")

# inputing data
cgpa = st.sidebar.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)

dsa = st.sidebar.number_input(
    "DSA Test Score(out of 100)", min_value=0, max_value=100, value=80
)

projects = st.sidebar.number_input(
    "Projects completed", min_value=0, max_value=20, value=5
)

internship = st.sidebar.number_input(
    "Internships completed", min_value=0, max_value=6, value=1
)

communication = st.sidebar.number_input(
    "Communication Test score( out of 100 )", min_value=0, max_value=100, value=80
)

student_details = [cgpa, projects, dsa, internship, communication]

input_array = np.array(student_details).reshape(1, -1)

st.markdown("---")
st.subheader("📌 Enter details from the sidebar and click Predict")

# output showing
if st.sidebar.button("Predict Package"):
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    st.success(f"🎉Estimated Package: {prediction[0]:.2f} LPA")

    st.info("Note: This prediction is based on a Multiple Linear Regression model.")


st.markdown("---")
st.subheader("ℹ️ Model Information")
st.write("Algorithm Used: Multiple Linear Regression")
st.write("Interface Built With: Streamlit")
st.write("Features Used: CGPA, DSA Score, Projects, Internships, Communication Score")
