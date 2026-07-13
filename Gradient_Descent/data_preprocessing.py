import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("Gradient_Descent/data/student_marks_regression.csv")
    return df


def clean_data(df):

    print("Before Cleaning Rows")
    print("-" * 50)
    print(df.shape[0])
    df = df.drop_duplicates()
    print("\nRows After Cleaning")
    print("-" * 50)
    print(df.shape[0])
    return df


def train_test(df):

    X = df["StudyHours"]
    Y = df["Marks"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=56
    )

    print("\nAfter Splitting")
    print("-" * 50)
    print(f"X_Train Size = {X_train.shape}")
    print(f"X_Test Size = {X_test.shape}")
    print(f"Y_Train Size = {Y_train.shape}")
    print(f"Y_Test Size = {Y_test.shape}")


def main():
    df = load_data()
    df = clean_data(df)
    train_test(df)


if __name__ == "__main__":
    main()
