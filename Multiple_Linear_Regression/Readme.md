# Multiple Linear Regression From Scratch

This project is my implementation of Multiple Linear Regression using Python. After understanding Simple Linear Regression, I wanted to learn how regression works when multiple features contribute to predicting a single target value. Instead of directly using Scikit-learn, I first implemented the mathematical solution from scratch using the Normal Equation and then compared the results with Scikit-learn's implementation.

The model predicts a student's placement package based on multiple factors such as CGPA, number of projects, DSA score, internships, and communication skills.

## Project Structure

```text
Multiple_Linear_Regression/
│
├── dataset/
│   └── placement_package_regression.csv
│
├── scratch_mlr.py
├── sklearn_mlr.py
├── comparison.py
├── README.md
└── requirements.txt
```

## What's Included

- Multiple Linear Regression implemented from scratch
- Scikit-learn implementation for comparison
- Performance evaluation using standard regression metrics
- User input for package prediction
- Comparison between both implementations

## Features Used

- CGPA
- Projects
- DSA Score
- Internships
- Communication Score

## Target

- Placement Package (LPA)

## Evaluation Metrics

Both models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## What I Learned

This project helped me understand how Multiple Linear Regression extends the concept of Simple Linear Regression by working with multiple input features. While implementing the algorithm from scratch, I learned how the Normal Equation is used to calculate model parameters, how feature matrices are handled, and how predictions are generated using learned coefficients. Comparing the custom model with Scikit-learn also gave me confidence that my implementation was mathematically correct.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the scratch implementation:

```bash
python scratch_mlr.py
```

Run the Scikit-learn implementation:

```bash
python sklearn_mlr.py
```

Compare both models:

```bash
python comparison.py
```

## Future Improvements

- Implement Gradient Descent for Multiple Linear Regression
- Add Feature Scaling
- Compare Normal Equation with Gradient Descent
- Extend the project with Regularization techniques such as Ridge and Lasso Regression

## Author

**Giridhar Jadon**

B.Tech – Artificial Intelligence & Machine Learning

Aspiring AI/ML Engineer