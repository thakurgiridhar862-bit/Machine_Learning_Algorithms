import pandas as pd


def load_data():

    df = pd.read_csv(
        "Gradient_Descent/Batch_GD_Ndim/data/placement_package_regression.csv"
    )
    return df


def data_overview(df):

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nFirst Five Rows")
    print("-" * 50)
    print(df.head(5))

    print("\nDataset Size")
    print("-" * 50)
    print(f"Total Rows    : {df.shape[0]}")
    print(f"Total Columns : {df.shape[1]}")

    print("\nColumns Name")
    print("-" * 50)
    print(df.columns.tolist())

    print("\nData Types")
    print("-" * 60)
    print(df.dtypes)

    print("\nMissing Values")
    print("-" * 50)
    print(df.isnull().sum())
    print(f"Total Missing Values : {df.isnull().sum().sum()}")

    print("\nDUPLICATE VALUES")
    print("-" * 60)
    print(f"Total Duplicate Rows : {df.duplicated().sum()}")

    print("\nSTATISTICAL SUMMARY")
    print("-" * 60)
    print(df.describe())


def main():
    df = load_data()
    data_overview(df)


if __name__ == "__main__":
    main()
