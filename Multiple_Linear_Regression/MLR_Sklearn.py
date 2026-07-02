import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("Multiple_Linear_Regression/data/placement_package_regression.csv")

X = df[["CGPA", "Projects", "DSA_Score", "Internships", "Communication_Score"]]
Y = df["Package_LPA"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=56
)
lr = LinearRegression()
lr.fit(X_train, Y_train)

Cgpa = float(input("Enter Your Cgpa : "))
Pro = int(input("Enter the number of projects You have done : "))
DSA = int(input("Enter your DSA test score : "))
Intern = int(input("Enter the number of internships you have done : "))
Communication = int(input("Enter your comunication test score : "))

input_data = pd.DataFrame(
    [[Cgpa, Pro, DSA, Intern, Communication]],
    columns=["CGPA", "Projects", "DSA_Score", "Internships", "Communication_Score"],
)

y_pred = lr.predict(input_data)
print("Acc to my Prediction Your package will be : ", y_pred)

y_pred = lr.predict(X_test)

print("R² Score :", r2_score(Y_test, y_pred))
print("MAE      :", mean_absolute_error(Y_test, y_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, y_pred)))
print("Slope values : ", lr.coef_)
print("Intercept : ", lr.intercept_)
