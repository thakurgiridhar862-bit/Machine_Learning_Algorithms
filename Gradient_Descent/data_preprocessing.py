import pandas as pd


def load_data():
    df = pd.read_csv("Gradient_Descent/data/student_marks_regression.csv")
    return df


def main():
    df = load_data()


if __name__ == "__main__":
    main()
