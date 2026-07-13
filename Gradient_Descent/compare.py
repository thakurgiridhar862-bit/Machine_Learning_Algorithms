from data_preprocessing import load_data, clean_data, train_test
from sklearn.linear_model import LinearRegression


def main():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, Y_train, Y_test = train_test(df)

    X_train_2d = X_train.values.reshape(-1, 1)
    X_test_2d = X_test.values.reshape(-1, 1)


if __name__ == "__main__":
    main()
