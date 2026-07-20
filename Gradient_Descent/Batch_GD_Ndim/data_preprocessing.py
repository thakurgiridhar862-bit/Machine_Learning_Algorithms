from data_overview import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def tts(df):

    X = df[
        [
            "CGPA",
            "Projects",
            "DSA_Score",
            "Internships",
            "Communication_Score",
        ]
    ]

    Y = df["Package_LPA"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=56,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nAfter Splitting")
    print("-" * 50)
    print(f"X Train Size = {X_train.shape}")
    print(f"X Test Size  = {X_test.shape}")
    print(f"Y Train Size = {Y_train.shape}")
    print(f"Y Test Size  = {Y_test.shape}")

    print("\nAfter Feature Scaling")
    print("-" * 50)
    print(f"X Train Scaled Size = {X_train_scaled.shape}")
    print(f"X Test Scaled Size  = {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler


def main():
    df = load_data()

    X_train, X_test, Y_train, Y_test, scaler = tts(df)


if __name__ == "__main__":
    main()
