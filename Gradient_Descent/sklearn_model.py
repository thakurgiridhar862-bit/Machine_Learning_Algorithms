from data_preprocessing import load_data, clean_data, train_test
from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    df = load_data()

    df = clean_data(df)
    X_train, X_test, Y_train, Y_test = train_test(df)

    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)

    comparision = pd.DataFrame(
        {
            "StudyHours": X_test.reset_index(drop=True),
            "Marks": Y_test.reset_index(drop=True),
            "Predicted Marks": Y_pred.reset_index(drop=True),
        }
    )
