import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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


X = df[["CGPA", "Projects", "DSA_Score", "Internships", "Communication_Score"]]
Y = df["Package_LPA"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=56
)
scratch_model = IshiMLR()
scratch_model.fit(X_train, Y_train)
scratch_pred = scratch_model.predict(X_test)

sklearn_model = LinearRegression()
sklearn_model.fit(X_train, Y_train)
sklearn_pred = sklearn_model.predict(X_test)

print("=" * 60)
print("MULTIPLE LINEAR REGRESSION MODEL COMPARISON")
print("=" * 60)

print("\nScratch Model Results")
print("-" * 60)
print("R² Score :", r2_score(Y_test, scratch_pred))
print("MAE      :", mean_absolute_error(Y_test, scratch_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, scratch_pred)))
print("Slope    :", scratch_model.coef)
print("Intercept:", scratch_model.intercept)

print("\nScikit-learn Model Results")
print("-" * 60)
print("R² Score :", r2_score(Y_test, sklearn_pred))
print("MAE      :", mean_absolute_error(Y_test, sklearn_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, sklearn_pred)))
print("Slope    :", sklearn_model.coef_[0])
print("Intercept:", sklearn_model.intercept_)

comparison = pd.DataFrame(
    {
        "Actual": Y_test.values,
        "Scratch_Predicted": scratch_pred,
        "Sklearn_Predicted": sklearn_pred,
    }
)

print("\nActual vs Predicted")
print("-" * 60)
print(comparison.head(10))
