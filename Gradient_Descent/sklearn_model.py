from data_preprocessing import load_data, clean_data, train_test
from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, Y_train, Y_test = train_test(df)

    X_train_2d = X_train.values.reshape(-1, 1)
    X_test_2d = X_test.values.reshape(-1, 1)

    lr = LinearRegression()

    lr.fit(X_train_2d, Y_train)

    Y_pred = lr.predict(X_test_2d)

    comparison = pd.DataFrame(
        {
            "StudyHours": X_test.reset_index(drop=True),
            "Actual Marks": Y_test.reset_index(drop=True),
            "Predicted Marks": Y_pred,
        }
    )

    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = mse**0.5
    r2 = r2_score(Y_test, Y_pred)

    print("\nFINAL MODEL VALUES")
    print("-" * 50)
    print(f"Slope     : {lr.coef_[0]}")
    print(f"Intercept : {lr.intercept_}")

    print("\nACTUAL VS PREDICTED")
    print("-" * 50)
    print(comparison.head(10))

    print("\nMODEL EVALUATION")
    print("-" * 50)
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")


if __name__ == "__main__":
    main()
