import pandas as pd


def load_data():

    df = pd.read_csv(
        "Gradient_Descent/Batch_GD_Ndim/data/placement_package_regression.csv"
    )
    return df


def main():
    df = load_data()


if __name__ == "__main__":
    main()
