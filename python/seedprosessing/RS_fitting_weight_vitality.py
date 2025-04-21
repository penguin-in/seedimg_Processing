import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def logistic_regression_analysis(weight_data, binary_data):

    # Convert to DataFrame
    df = pd.DataFrame({
        'Weight': weight_data,
        'Target': binary_data
    })

    # Prepare data
    X = df[['Weight']]  # Feature matrix
    y = df['Target']  # Target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LogisticRegression(penalty='none')  # Disable regularization
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'coefficients': {
            'intercept': model.intercept_[0],
            'weight': model.coef_[0][0]
        }
    }

    # Visualization
    plt.figure(figsize=(12, 5))

    # Plot 1: Decision boundary
    plt.subplot(1, 2, 1)
    plt.scatter(X_test, y_test, c=y_prob, cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Predicted Probability')
    plt.xlabel('primeter')
    plt.ylabel('Target Class')
    plt.title('Classification Results')

    # Plot 2: Sigmoid curve
    plt.subplot(1, 2, 2)
    weight_range = np.linspace(X.min(), X.max(), 300)
    probas = model.predict_proba(weight_range.reshape(-1, 1))[:, 1]
    plt.plot(weight_range, probas, color='darkorange', lw=2)
    plt.xlabel('primeter')
    plt.ylabel('Probability of Class 1')
    plt.title('Logistic Regression Curve')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Rs_fitting_weight_vitality_distribution", dpi=600)
    plt.show()

    return model, results


# Example usage
if __name__ == "__main__":
    # Generate sample data
    excel_file_path = 'sorted_output.xlsx'
    data_path = 'entire_data.xlsx'

    data0 = pd.read_excel(data_path, sheet_name='Sheet1')
    data0 = np.array(data0, dtype=np.float64)
    all_sheets = pd.read_excel(excel_file_path, sheet_name=None, header=None)
    ori_data = np.concatenate((all_sheets['2'], all_sheets['3'], all_sheets['4'],
                               all_sheets['5'], all_sheets['6'], all_sheets['7'],
                               all_sheets['8'], all_sheets['9'], all_sheets['10'],
                               all_sheets['11']), axis=0)
    data = np.concatenate((ori_data[:, 1:8], ori_data[:, 11].reshape(-1, 1)), axis=1)
    data = np.array(data, dtype=np.float64)


    data_vitality = data[:, 7]
    weight = data[:, 0]
    area = data0[:, 8]
    primeter = data0[:, 9]
    # Run analysis
    model, results = logistic_regression_analysis(primeter, data_vitality)

    # Print results
    print("=== Model Coefficients ===")
    print(f"Intercept: {results['coefficients']['intercept']:.4f}")
    print(f"Weight Coefficient: {results['coefficients']['weight']:.4f}")

    print("\n=== Model Performance ===")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"ROC AUC: {results['roc_auc']:.3f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])