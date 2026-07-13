import pandas as pd


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


def main():
    df = load_data()
    df = clean_data(df)


if __name__ == "__main__":
    main()
