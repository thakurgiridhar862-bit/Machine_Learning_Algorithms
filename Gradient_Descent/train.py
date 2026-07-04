import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("Gradient_Descent/data/student_marks_regression.csv")
    return df


def data_overview(df):
    print("-" * 60)
    print("DATASET OVERVIEW")
    print("-" * 50)

    print("\nFIRST FIVE ROWS")
    print(df.head(5))

    print(f"\nTotal row              : {df.shape[0]}")
    print(f"Total Columns          : {df.shape[1]}")
    print(f"Total Null Values      : {df.isnull().sum().sum()}")
    print(f"Total Duplicate Values : {df.duplicated().sum()}")

    print("\nMISSING VALUES ")
    print("-" * 50)
    print(df.isnull().sum())

    print(" \n DATA TYPES")
    print("-" * 50)
    print(df.dtypes)

    print("\nNUMERICAL SUMMARY")
    print("-" * 50)
    print(df.describe())


def feature_selection(df):
    X = df[["StudyHours"]]
    Y = df["Marks"]

    print("\nFEATURE SELECTION")
    print("-" * 50)

    print("\nInput Feature (X):")
    print(X.columns.tolist())

    print("\nTarget Variable (Y):")
    print(Y.name)

    return X, Y


def tts(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=56
    )
    print("\nTRAIN TEST SPLIT OF DATA")
    print("-" * 50)
    print(f"Shape of X_train : {X_train.shape}")
    print(f"Shape of X_test  : {X_test.shape}")
    print(f"Shape of Y_train : {Y_train.shape}")
    print(f"Shape of Y_test  : {Y_test.shape}")
    return X_train, X_test, Y_train, Y_test


def main():
    df = load_data()
    data_overview(df)
    X, Y = feature_selection(df)
    X_train, X_test, Y_train, Y_test = tts(X, Y)


if __name__ == "__main__":
    main()
