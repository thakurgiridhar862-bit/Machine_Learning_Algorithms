import matplotlib.pyplot as plt
from data_preprocessing import load_data, clean_data, train_test
from GD import GS_reg


def main():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, Y_train, Y_test = train_test(df)

    gs = GS_reg(0.00001, 1000)
    gs.fit(X_train, Y_train)

    Y_pred = gs.predict(X_test)

    X_line = df["StudyHours"].sort_values()
    Y_line = gs.predict(X_line)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["StudyHours"], df["Marks"], label="Actual Data")
    plt.plot(X_line, Y_line, label="Regression Line", color="red")
    plt.title("Study Hours vs Marks")
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "Gradient_Descent/graphs/regression_line.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(Y_test, Y_pred)
    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        linestyle="--",
        label="Ideal Prediction",
        color="red",
    )

    plt.legend()
    plt.title("Actual vs Predicted Marks")
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.grid(True)
    plt.savefig(
        "Gradient_Descent/graphs/actual_vs_predicted.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
