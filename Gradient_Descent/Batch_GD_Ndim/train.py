from data_preprocessing import load_data, tts
from Batch_GD import GS_Reg

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    df = load_data()

    X_train, X_test, Y_train, Y_test, scaler = tts(df)

    gs = GS_Reg(0.01, 1000)

    gs.fit(X_train, Y_train)

    Y_pred = gs.predict(X_test)

    comparison = pd.DataFrame(
        {
            "Actual_Package": Y_test.reset_index(drop=True),
            "Predicted_Package": Y_pred,
        }
    )

    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = mse**0.5
    r2 = r2_score(Y_test, Y_pred)

    print("\nFINAL MODEL VALUES")
    print("-" * 50)
    print(f"Coefficients : {gs.co}")
    print(f"Intercept    : {gs.inter}")

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
