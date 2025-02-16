from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torch
import sys
import numpy as np

sys.path.append("./")

from configs.main_config import CLASS_WEIGHTS

# 📌 Helper Function to Compute Metrics with Class Weights
def compute_metrics(y_true, y_pred_probs,threshold=0.5):
    """
    Compute loss, accuracy, precision, recall, and ROC-AUC with class weights.
    
    Args:
    - y_true (Tensor): True labels (shape: [batch_size, 1])
    - y_pred_probs (Tensor): Predicted probabilities/logits (shape: [batch_size, 1])
    - criterion: Loss function (e.g., BCEWithLogitsLoss with reduction='none')
    - class_weights (Tensor): Class weights (tensor of shape [2] -> [weight_for_0, weight_for_1])
    - threshold (float): Threshold for classification (default: 0.5)
    
    Returns:
    - Dictionary containing computed weighted metrics
    """

    y_true = np.array(y_true).flatten()
    y_pred_probs = np.array(y_pred_probs).flatten()

    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs > threshold) * 1
    y_pred = y_pred.flatten()

    sample_weights_np = np.where(y_pred == 1, CLASS_WEIGHTS[1], 1)  # Convert weights to NumPy

    # Compute Metrics with sample weights
    accuracy = accuracy_score(y_true, y_pred)  # Accuracy is unweighted
    precision = precision_score(y_true, y_pred, sample_weight=sample_weights_np, zero_division=0)
    recall = recall_score(y_true, y_pred, sample_weight=sample_weights_np, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred, sample_weight=sample_weights_np)  # Using probabilities for AUC

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }
