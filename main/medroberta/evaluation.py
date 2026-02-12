import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


def eval_regression(y_true, y_pred, name):
    """
    Evaluate regression predictions with RMSE, MAE, and a scatter plot.

    Parameters
    ----------
    y_true : array-like
        True numeric values
    y_pred : array-like
        Predicted numeric values
    name : str
        Name of the target variable or dataset
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Print metrics
    print(f"\nRegression: {name}")
    print(f"  - RMSE: {mean_squared_error(y_true, y_pred, squared=False):.4f}")
    print(f"  - MAE : {mean_absolute_error(y_true, y_pred):.4f}")

    # Scatter plot: true vs predicted
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # diagonal
    plt.title(f"{name} — True vs Predicted")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.show()


def eval_classification(y_true, y_pred, name, encoder=None):
    """
    Evaluate classification predictions with accuracy, F1, precision, recall, and optional confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    name : str
        Name of the classification task or dataset
    encoder : sklearn.preprocessing.LabelEncoder, optional
        If provided, used to map labels to class names
    """
    print(f"\nClassification: {name}")
    print(f"  - Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  - F1-score : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  - Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  - Recall   : {recall_score(y_true, y_pred, average='macro'):.4f}")

    # Detailed classification report
    if encoder:
        all_labels = np.arange(len(encoder.classes_))
        print(classification_report(
            y_true, y_pred,
            labels=all_labels,
            target_names=encoder.classes_,
            zero_division=0
        ))
    else:
        print(classification_report(y_true, y_pred))

    # Confusion matrix if classes are <= 20
    if len(set(y_true)) <= 20:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} — Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()


def plot_evaluation_results(classification_results, numeric_results, set_name='Train'):
    """
    Plot evaluation metrics for classification and numeric predictions side by side.

    Parameters
    ----------
    classification_results : dict
        Dictionary with keys=labels and values={'accuracy': float, 'f1': float, ...}
    numeric_results : dict
        Dictionary with keys=labels and values={'accuracy': float, 'mae': float, ...}
    set_name : str, optional
        Name of the dataset or split, e.g., 'Train' or 'Test' (default 'Train')
    """
    # Classification metrics plot
    if classification_results:
        labels_class = list(classification_results.keys())
        acc_values = [classification_results[l]['accuracy'] for l in labels_class]
        f1_values = [classification_results[l]['f1'] for l in labels_class]

        plt.figure(figsize=(8, 5))
        x = range(len(labels_class))
        plt.bar(x, acc_values, width=0.4, label='Accuracy', align='center')
        plt.bar([i + 0.4 for i in x], f1_values, width=0.4, label='F1-score', align='center')
        plt.xticks([i + 0.2 for i in x], labels_class)
        plt.ylabel("Score")
        plt.title(f"Classification Metrics per Label ({set_name})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Numeric metrics plot
    if numeric_results:
        labels_num = list(numeric_results.keys())
        mae_values = [numeric_results[l]['mae'] for l in labels_num]
        acc_num_values = [numeric_results[l]['accuracy'] for l in labels_num]

        plt.figure(figsize=(10, 5))
        x = range(len(labels_num))
        plt.bar(x, acc_num_values, width=0.4, label='Accuracy', align='center')
        plt.bar([i + 0.4 for i in x], mae_values, width=0.4, label='MAE', align='center')
        plt.xticks([i + 0.2 for i in x], labels_num, rotation=30, ha='right')
        plt.ylabel("Score")
        plt.title(f"Numeric Metrics per Label ({set_name})")
        plt.legend()
        plt.tight_layout()
        plt.show()
