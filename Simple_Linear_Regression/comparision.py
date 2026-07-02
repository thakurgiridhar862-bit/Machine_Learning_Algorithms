import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class IshiLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, Y_train):
        numerator = 0
        denominator = 0

        x_mean = X_train.mean()
        y_mean = Y_train.mean()

        for i in range(X_train.shape[0]):
            numerator += (Y_train.iloc[i] - y_mean) * (X_train.iloc[i] - x_mean)
            denominator += (X_train.iloc[i] - x_mean) ** 2

        self.m = numerator / denominator
        self.b = y_mean - (self.m * x_mean)

    def predict(self, X):
        return self.m * X + self.b


df = pd.read_csv("Simple_Linear_Regression/data/student_marks_regression.csv")

X = df["StudyHours"]
Y = df["Marks"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=56
)

scratch_model = IshiLR()
scratch_model.fit(X_train, Y_train)
scratch_pred = scratch_model.predict(X_test)

sklearn_model = LinearRegression()
sklearn_model.fit(X_train.values.reshape(-1, 1), Y_train)
sklearn_pred = sklearn_model.predict(X_test.values.reshape(-1, 1))

print("=" * 60)
print("SIMPLE LINEAR REGRESSION MODEL COMPARISON")
print("=" * 60)

print("\nScratch Model Results")
print("-" * 60)
print("R² Score :", r2_score(Y_test, scratch_pred))
print("MAE      :", mean_absolute_error(Y_test, scratch_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, scratch_pred)))
print("Slope    :", scratch_model.m)
print("Intercept:", scratch_model.b)

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
        "Scratch_Predicted": scratch_pred.values,
        "Sklearn_Predicted": sklearn_pred,
    }
)

print("\nActual vs Predicted")
print("-" * 60)
print(comparison.head(10))
