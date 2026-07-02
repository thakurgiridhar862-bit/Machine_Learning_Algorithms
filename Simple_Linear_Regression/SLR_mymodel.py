import pandas as pd

from sklearn.model_selection import train_test_split


# Load Dataset
df = pd.read_csv("Simple_Linear_Regression/data/student_marks_regression.csv")


class IshiLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, Y_train):
        num = 0
        den = 0

        x_mean = X_train.mean()
        y_mean = Y_train.mean()

        for i in range(X_train.shape[0]):
            num += (Y_train.iloc[i] - y_mean) * (X_train.iloc[i] - x_mean)
            den += (X_train.iloc[i] - x_mean) ** 2

        self.m = num / den
        self.b = y_mean - (self.m * x_mean)

    def predict(self, X):
        return self.m * X + self.b


X = df["StudyHours"]
Y = df["Marks"]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=56,
)

# Train Model
model = IshiLR()
model.fit(X_train, Y_train)

# Model Parameters
print("\nModel Parameters")
print("Slope     :", round(model.m, 4))
print("Intercept :", round(model.b, 4))

# User Prediction
hours = float(input("\nEnter Study Hours: "))

prediction = model.predict(hours)

print(f"\nPredicted Marks: {prediction:.2f}")
