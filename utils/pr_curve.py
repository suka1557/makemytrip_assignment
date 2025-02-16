import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
import sys, os

sys.path.append("./")

from configs.main_config import PR_PLOT_NAME, PLOTS_FOLDER

def plot_precision_recall_get_optimal_threshold(y_true, y_pred_probs, sample_weights, epoch, save_path=PR_PLOT_NAME, save_plot=False):
    """
    Plots and saves the Precision-Recall curve, then returns the best threshold.

    Args:
    - y_true (array-like): True binary labels.
    - y_pred_probs (array-like): Predicted probabilities (not binary).
    - save_path (str): File path to save the curve as a PNG.

    Returns:
    - best_threshold (float): Threshold where Precision ≈ Recall.
    """

    save_path = save_path + str(epoch) + ".png"
    save_path = os.path.join(PLOTS_FOLDER, save_path)

    # Convert Tensors to NumPy for sklearn metrics
    # y_true_np = y_true.detach().cpu().numpy()
    # y_pred_np = y_pred.detach().cpu().numpy()
    # y_pred_probs_np = y_pred_probs.detach().cpu().numpy()
    # sample_weights_np = sample_weights.detach().cpu().numpy()  # Convert weights to NumPy

    # Compute Precision-Recall values
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs, pos_label=1)

    # Find the best threshold (where Precision ≈ Recall)
    best_index = np.argmin(np.abs(precisions - recalls))
    best_threshold = thresholds[best_index]
    best_precision = precisions[best_index]
    best_recall = recalls[best_index]

    if save_plot:

        # Plot Precision-Recall Curve (Recall on X, Precision on Y)
        plt.plot(recalls, precisions, label="Precision-Recall Curve", linestyle="-", color="b")

        # 45-degree line (y = x) for reference
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Model (y=x)")

        # Mark the best threshold point
        plt.scatter(best_recall, best_precision, color="red", s=100, label=f"Best Threshold: {best_threshold:.2f}")

        # Labels and Title
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()

        # Save the plot
        plt.savefig(save_path)
        plt.close()  # Close the plot to free memory


    return best_threshold
