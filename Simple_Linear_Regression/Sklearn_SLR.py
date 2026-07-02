import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load dataset
df = pd.read_csv("Simple_Linear_Regression/data/student_marks_regression.csv")

X = df[["StudyHours"]]
Y = df["Marks"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=56
)

# Model training
model = LinearRegression()
model.fit(X_train, Y_train)

# Prediction
Y_pred = model.predict(X_test)


# Evaluation
print("\nModel Evaluation")
print("R² Score :", r2_score(Y_test, Y_pred))
print("MAE      :", mean_absolute_error(Y_test, Y_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, Y_pred)))

print("\nModel Parameters")
print("Slope     :", model.coef_[0])
print("Intercept :", model.intercept_)

# Visualization
plt.figure(figsize=(10, 6))
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True, alpha=0.5)

sns.scatterplot(data=df, x="StudyHours", y="Marks", alpha=0.7)

X_line = df.sort_values("StudyHours")

plt.plot(X_line["StudyHours"], model.predict(X_line[["StudyHours"]]), color="red")

plt.show()
