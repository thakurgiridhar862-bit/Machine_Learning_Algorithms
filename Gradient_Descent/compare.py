from data_preprocessing import load_data, clean_data, train_test
from GD import GS_reg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np


def main():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, Y_train, Y_test = train_test(df)

    X_train_2d = X_train.values.reshape(-1, 1)
    X_test_2d = X_test.values.reshape(-1, 1)

    gs = GS_reg(0.00001, 1000)

    gs.fit(X_train, Y_train)

    Y_pred_GS = gs.predict(X_test)

    lr = LinearRegression()

    lr.fit(X_train_2d, Y_train)

    Y_pred_lr = lr.predict(X_test_2d)

    print("=" * 70)
    print("SIMPLE LINEAR REGRESSION Vs GRADIENT DESCENT MODEL COMPARISON")
    print("=" * 70)

    print("\nGradient Descent Model Results")
    print("-" * 60)
    print("R² Score :", r2_score(Y_test, Y_pred_GS))
    print("MAE      :", mean_absolute_error(Y_test, Y_pred_GS))
    print("RMSE     :", np.sqrt(mean_squared_error(Y_test, Y_pred_GS)))
    print("Slope    :", gs.m)
    print("Intercept:", gs.b)

    print("\nScikit-learn Model Results")
    print("-" * 60)
    print("R² Score :", r2_score(Y_test, Y_pred_lr))
    print("MAE      :", mean_absolute_error(Y_test, Y_pred_lr))
    print("RMSE     :", np.sqrt(mean_squared_error(Y_test, Y_pred_lr)))
    print("Slope    :", lr.coef_[0])
    print("Intercept:", lr.intercept_)


if __name__ == "__main__":
    main()
