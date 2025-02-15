from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torch

# 📌 Helper Function to Compute Metrics with Class Weights
def compute_metrics(y_true, y_pred_probs, criterion, class_weights, threshold=0.5):
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
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs > threshold).float()

    # Compute per-sample class weights based on true labels
    sample_weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]

    # Compute Weighted Loss
    per_sample_loss = criterion(y_pred_probs, y_true)
    weighted_loss = (per_sample_loss * sample_weights).mean().item()  # Apply weights and average

    # Convert Tensors to NumPy for sklearn metrics
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_probs_np = y_pred_probs.detach().cpu().numpy()
    sample_weights_np = sample_weights.detach().cpu().numpy()  # Convert weights to NumPy

    # Compute Metrics with sample weights
    accuracy = accuracy_score(y_true_np, y_pred_np)  # Accuracy is unweighted
    precision = precision_score(y_true_np, y_pred_np, sample_weight=sample_weights_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, sample_weight=sample_weights_np, zero_division=0)
    roc_auc = roc_auc_score(y_true_np, y_pred_probs_np, sample_weight=sample_weights_np)  # Using probabilities for AUC

    return {
        "loss": weighted_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }
