import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("Placement_Package_Predictor/data/placement_package_regression.csv")


class IshiMLR:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X_train, Y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)

        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(Y_train)

        self.coef = betas[1:]
        self.intercept = betas[0]

    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef) + self.intercept
        return y_pred


# Formattiong data
X = df[["CGPA", "Projects", "DSA_Score", "Internships", "Communication_Score"]]
Y = df["Package_LPA"]

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=56
)

model = IshiMLR()
model.fit(X_train, Y_train)


# Input data
Cgpa = float(input("Enter Your Cgpa : "))
Pro = int(input("Enter the number of projects You have done : "))
DSA = int(input("Enter your DSA test score : "))
Intern = int(input("Enter the number of internships you have done : "))
Communication = int(input("Enter your comunication test score : "))

input_data = pd.DataFrame(
    [[Cgpa, Pro, DSA, Intern, Communication]],
    columns=["CGPA", "Projects", "DSA_Score", "Internships", "Communication_Score"],
)

y_pred = model.predict(input_data)
print("Acc to my Prediction Your package will be : ", y_pred)

Y_pred = model.predict(X_test)

print("R² Score :", r2_score(Y_test, Y_pred))
print("MAE      :", mean_absolute_error(Y_test, Y_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, Y_pred)))
print("Coef     :", model.coef)
print("Intercept:", model.intercept)
