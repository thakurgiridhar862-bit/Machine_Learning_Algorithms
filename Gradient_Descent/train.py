from data_preprocessing import load_data, clean_data, train_test
from GD import GS_reg


def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, Y_train, Y_test = train_test(df)


if __name__ == "__main__":
    main()
