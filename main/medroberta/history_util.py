from typing import Dict, Any, List
import numpy as np


def _safe_mean(values: List[float]) -> float:
    """
    Compute the mean of a list safely, ignoring NaNs.
    Returns NaN if the list is empty.

    Parameters
    ----------
    values : List[float]
        List of numeric values

    Returns
    -------
    float
        Mean value or NaN
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float('nan')
    return float(np.nanmean(arr))


def compute_model_averages(trained_models: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute average metrics for each trained model over epochs.

    For classification models:
        - avg_accuracy
        - avg_macro_f1
        - avg_r, avg_p (optional)
        - last_accuracy, last_macro_f1 (from final epoch)
    For regression models:
        - avg_rmse
        - avg_mae
        - last_rmse, last_mae (from final epoch)

    Always includes:
        - avg_loss
        - num_epochs

    Parameters
    ----------
    trained_models : Dict[str, Dict[str, Any]]
        Dictionary mapping label/target name to a bundle containing
        history of metrics and task_type ('classification' or 'regression').

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with computed averages per model
    """
    out: Dict[str, Dict[str, float]] = {}

    for label_col, bundle in trained_models.items():
        hist = (bundle or {}).get('history', {})
        task_type = (bundle or {}).get('task_type', 'classification')

        avg_loss = _safe_mean(hist.get('loss', []))
        num_epochs = len(hist.get('loss', []))

        stats = {'avg_loss': avg_loss, 'num_epochs': float(num_epochs)}

        if task_type == 'classification':
            stats['avg_accuracy'] = _safe_mean(hist.get('accuracy', []))
            stats['avg_macro_f1'] = _safe_mean(hist.get('macro_f1', []))
            stats['avg_r'] = _safe_mean(hist.get('r', []))
            stats['avg_p'] = _safe_mean(hist.get('p', []))

            if hist.get('accuracy'):
                stats['last_accuracy'] = float(hist['accuracy'][-1])
            if hist.get('macro_f1'):
                stats['last_macro_f1'] = float(hist['macro_f1'][-1])
        else:
            stats['avg_rmse'] = _safe_mean(hist.get('rmse', []))
            stats['avg_mae'] = _safe_mean(hist.get('mae', []))

            if hist.get('rmse'):
                stats['last_rmse'] = float(hist['rmse'][-1])
            if hist.get('mae'):
                stats['last_mae'] = float(hist['mae'][-1])

        out[label_col] = stats

    return out


def plot_loss_curves(trained_models: Dict[str, Dict[str, Any]]):
    """
    Plot training loss curves for multiple models in a single figure.

    Each model's loss history is plotted as a separate line.

    Parameters
    ----------
    trained_models : Dict[str, Dict[str, Any]]
        Dictionary mapping label/target name to a bundle containing
        history of metrics including 'loss'
    """
    import matplotlib.pyplot as plt

    plt.figure()
    has_any = False

    for label_col, bundle in trained_models.items():
        hist = (bundle or {}).get('history', {})
        losses = hist.get('loss', [])
        if losses:
            epochs = list(range(1, len(losses) + 1))
            plt.plot(epochs, losses, label=str(label_col))
            has_any = True

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Model')
    if has_any:
        plt.legend()
    plt.tight_layout()
    plt.show()
