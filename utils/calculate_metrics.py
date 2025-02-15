from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 📌 Helper Function to Compute Metrics
def compute_metrics(y_true, y_pred_probs, criterion):
    """
    Compute loss, accuracy, precision, recall, and ROC-AUC.
    
    Args:
    - y_true (Tensor): True labels
    - y_pred_probs (Tensor): Predicted probabilities
    - criterion: Loss function (e.g., BCEWithLogitsLoss)
    
    Returns:
    - Dictionary containing computed metrics
    """
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_probs > 0.5).float()

    # Compute Loss
    loss = criterion(y_pred_probs, y_true).item()

    # Convert to NumPy for sklearn metrics
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_pred_probs_np = y_pred_probs.cpu().numpy()

    # Compute Metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, zero_division=0)
    roc_auc = roc_auc_score(y_true_np, y_pred_probs_np)  # Using probabilities for AUC

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }
