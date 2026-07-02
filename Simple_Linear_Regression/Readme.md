# 🚀 Simple Linear Regression From Scratch

> A complete implementation of **Simple Linear Regression from scratch using Python**, followed by a comparison with Scikit-learn's implementation to validate the mathematical correctness of the algorithm.

---

## 📌 Project Overview

Machine Learning libraries make training models extremely easy, but understanding **what happens behind the scenes** is equally important.

This project focuses on implementing **Simple Linear Regression** from scratch using mathematical formulas without relying on Scikit-learn's regression algorithm.

After implementing the algorithm manually, the results are compared against **Scikit-learn's LinearRegression** model to verify the correctness of the implementation.

---

## 🎯 Objective

- Understand the mathematics behind Linear Regression.
- Implement the algorithm completely from scratch.
- Predict student marks based on study hours.
- Compare the custom implementation with Scikit-learn.
- Validate the implementation using standard evaluation metrics.

---

## 📂 Dataset

The dataset contains two features:

| Feature | Description |
|----------|-------------|
| StudyHours | Number of hours studied |
| Marks | Student's obtained marks |

---

## 📁 Project Structure

```
Simple_Linear_Regression
│
├── dataset/
│   └── student_marks_regression.csv
│
├── sklearn_slr.py
├── scratch_slr.py
├── comparison.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Implementations Included

### ✅ Scratch Implementation

Implemented manually using:

- Mean
- Covariance
- Variance
- Slope Calculation
- Intercept Calculation
- Prediction Formula

No Machine Learning regression library is used for training.

---

### ✅ Scikit-learn Implementation

Implemented using

- LinearRegression
- Train-Test Split
- Model Evaluation

to compare the custom model against an industry-standard implementation.

---

## 📈 Model Evaluation

Both implementations are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The comparison demonstrates that the scratch implementation produces results very close to Scikit-learn's model.

---

## 📊 Workflow

```
Dataset
    │
    ▼
Train-Test Split
    │
    ▼
Scratch Model
    │
    ├── Calculate Mean
    ├── Calculate Slope
    ├── Calculate Intercept
    └── Predict
    │
    ▼
Evaluation
    │
    ▼
Comparison with Scikit-learn
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 💻 How to Run

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Scratch Model

```bash
python scratch_slr.py
```

Run Scikit-learn Model

```bash
python sklearn_slr.py
```

Compare Both Models

```bash
python comparison.py
```

---

## 🎓 Learning Outcomes

This project helped strengthen my understanding of:

- Machine Learning Fundamentals
- Linear Regression Mathematics
- Regression Intuition
- Model Evaluation
- Python Object-Oriented Programming
- Algorithm Implementation from Scratch

---

## 🚀 Future Improvements

- Gradient Descent Implementation
- Stochastic Gradient Descent
- Mini Batch Gradient Descent
- Polynomial Regression
- Ridge Regression
- Lasso Regression

---

## 👨‍💻 Author

**Giridhar Jadon**

B.Tech (Artificial Intelligence & Machine Learning)

Aspiring AI/ML Engineer

---

## ⭐ Repository Goal

This repository is a part of my **Machine Learning From Scratch** series, where every Machine Learning algorithm will first be implemented mathematically from scratch and then validated using Scikit-learn.

Upcoming Implementations:

- ✅ Simple Linear Regression
- ⏳ Multiple Linear Regression
- ⏳ Gradient Descent
- ⏳ Logistic Regression
- ⏳ Naive Bayes
- ⏳ K-Nearest Neighbors
- ⏳ Decision Trees
- ⏳ Random Forest
- ⏳ Support Vector Machine