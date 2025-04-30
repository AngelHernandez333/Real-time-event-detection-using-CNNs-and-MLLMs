import numpy as np

def calculate_ap(precision, recall):
    # Sort by recall (ascending)
    sorted_indices = np.argsort(recall)
    precision = np.array(precision)[sorted_indices]
    recall = np.array(recall)[sorted_indices]

    # Pad with (0,0) and (1,0)
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    # Compute AP as the area under the raw curve (no interpolation)
    ap = 0.0
    for i in range(1, len(recall)):
        delta_recall = recall[i] - recall[i-1]
        ap += delta_recall * precision[i]

    return ap

frame1=np.load
# Sample data: 20 frames from 2 videos, 5 anomalous
scores = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
        0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00])
ground_truth = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Get unique scores as thresholds (sorted descending)
thresholds = np.unique(scores)[::-1]

precision = []
recall = []

# Apply each threshold
for thresh in thresholds:
    # Predict 1 if score >= threshold, 0 otherwise
    predictions = (scores >= thresh).astype(int)
    
    # Compute TP, FP, FN
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    # Calculate precision and recall
    prec = tp / (tp + fp + 1e-10)  # Avoid division by zero
    rec = tp / (tp + fn + 1e-10)
    
    precision.append(prec)
    recall.append(rec)

# Compute AP
ap = calculate_ap(precision, recall)
print(f"Average Precision (AP): {ap:.4f}")

# Optional: Print precision-recall pairs for inspection
for t, p, r in zip(thresholds, precision, recall):
    print(f"Threshold: {t:.2f}, Precision: {p:.4f}, Recall: {r:.4f}")