# Simple Linear Regression From Scratch

This project is my implementation of Simple Linear Regression using Python. Instead of directly using Scikit-learn for training, I first built the algorithm from scratch to understand how linear regression actually works behind the scenes. After completing the custom implementation, I compared it with Scikit-learn's implementation to verify that both models produce nearly identical results.

The project uses a simple dataset where the goal is to predict a student's marks based on the number of study hours.

## Project Structure

```text
Simple_Linear_Regression/
│
├── dataset/
│   └── student_marks_regression.csv
│
├── scratch_slr.py
├── sklearn_slr.py
├── comparison.py
├── README.md
└── requirements.txt
```

## What's Included

- A complete implementation of Simple Linear Regression from scratch
- Scikit-learn implementation for comparison
- Performance evaluation using standard regression metrics
- Visualization of the regression line
- Comparison between both models

## Evaluation Metrics

The models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## What I Learned

Working on this project helped me understand the mathematics behind Simple Linear Regression instead of treating it as a black-box algorithm. I learned how the slope and intercept are calculated, how predictions are generated, and how a custom implementation can be validated by comparing it with Scikit-learn. This project also strengthened my understanding of regression evaluation metrics and the complete workflow of building a basic machine learning model.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the scratch implementation:

```bash
python scratch_slr.py
```

Run the Scikit-learn implementation:

```bash
python sklearn_slr.py
```

Compare both models:

```bash
python comparison.py
```

## Author

**Giridhar Jadon**

B.Tech – Artificial Intelligence & Machine Learning

Aspiring AI/ML Engineer