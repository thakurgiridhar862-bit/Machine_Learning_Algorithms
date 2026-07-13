from data_preprocessing import load_data, clean_data, train_test
from GD import GS_reg
from sklearn.linear_model import LinearRegression


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


if __name__ == "__main__":
    main()
