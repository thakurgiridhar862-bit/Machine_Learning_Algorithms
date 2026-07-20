from data_overview import load_data
import pandas as pd
from sklearn.model_selection import train_test_split


def tts(df):

    X = df[["CGPA", "Projects", "DSA_Score", "Internships", "Communication_Score"]]
    Y = df["Package_LPA"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=56
    )

    print("\nAfter Splitting")
    print("-" * 50)
    print(f"X_Train Size = {X_train.shape}")
    print(f"X_Test Size = {X_test.shape}")
    print(f"Y_Train Size = {Y_train.shape}")
    print(f"Y_Test Size = {Y_test.shape}")
    return X_train, X_test, Y_train, Y_test


def main():
    df = load_data()
    X_train, X_test, Y_train, Y_test = tts(df)


if __name__ == "__main__":
    main()
