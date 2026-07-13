from data_preprocessing import load_data, clean_data, train_test
from GD import GS_reg

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, Y_train, Y_test = train_test(df)

    gs = GS_reg(0.00001, 1000)

    gs.fit(X_train, Y_train)

    Y_pred = gs.predict(X_test)

    comparison = pd.DataFrame(
        {
            "StudyHours": X_test.reset_index(drop=True),
            "Actual_Marks": Y_test.reset_index(drop=True),
            "Predicted_Marks": Y_pred.reset_index(drop=True),
        }
    )

    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = mse**0.5
    r2 = r2_score(Y_test, Y_pred)

    print("\nFINAL MODEL VALUES")
    print("-" * 50)
    print(f"Slope     : {gs.m}")
    print(f"Intercept : {gs.b}")

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
