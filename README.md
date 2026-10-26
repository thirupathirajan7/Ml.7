# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project uses machine learning to identify potentially fraudulent credit card transactions. A Random Forest Classifier is trained on transaction data to predict whether a transaction is genuine or fraudulent.

## Objectives
- Explore and understand the credit card transaction dataset.
- Check for missing values and examine the target class distribution.
- Separate the input features from the target column.
- Train a Random Forest classification model.
- Evaluate the model using classification metrics and visualizations.
- Review detected and missed fraud cases.

## Dataset
The notebook loads a CSV file named `creditcard.csv`.

The target column is `Class`:
- `0` — Genuine transaction
- `1` — Fraudulent transaction

Make sure `creditcard.csv` is available at the path expected by the notebook (`/content/creditcard.csv` in Google Colab).

## Technologies and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Workflow
1. Import the required Python libraries.
2. Load and inspect the dataset.
3. Check dataset information, dimensions, missing values, and class distribution.
4. Separate features (`X`) and target (`y`).
5. Remove rows containing missing values.
6. Split the data into training and testing sets, using an 80/20 split and stratification.
7. Train a Random Forest Classifier with 100 trees and class-weight balancing.
8. Predict transaction classes and fraud probabilities.
9. Evaluate the model using a confusion matrix, precision, recall, F1-score, ROC-AUC, and a classification report.
10. Visualize feature importance and the ROC curve.
11. Save actual fraudulent transactions from the test set to `actual_fraudulent_test_transactions.csv`.

## Model Evaluation
The notebook calculates the following metrics:

- **Precision:** How many transactions predicted as fraud were actually fraud.
- **Recall:** How many actual fraud cases the model detected.
- **F1-score:** The balance between precision and recall.
- **ROC-AUC:** How well the model distinguishes between genuine and fraudulent transactions.
- **Confusion Matrix:** Summarizes correct and incorrect predictions for both classes.

Actual metric values depend on the dataset and the execution results.

## How to Run
1. Open `ML_Task8.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `creditcard.csv` to the expected location, or update the file path in the notebook.
3. Install the required libraries if they are not already available:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Run the notebook cells from top to bottom.
5. Review the printed evaluation metrics and generated plots.

## Output
- Confusion matrix visualization
- Classification report and evaluation metrics
- List of actual fraudulent test transactions with predictions and probabilities
- `actual_fraudulent_test_transactions.csv` output file
- Top 10 important features plot
- ROC curve

## Conclusion
This project demonstrates how a Random Forest Classifier can be used for credit card fraud detection. It explores the dataset, trains a model, evaluates its predictions, and visualizes useful results. The model's performance should be assessed carefully because fraudulent transactions are typically much less common than genuine transactions.

## Disclaimer
This is an educational machine learning project. Its predictions should not be treated as a definitive decision for real financial transactions.
