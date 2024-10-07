import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch

# 1. Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x=[f'Fold {i+1}' for i in range(kf.n_splits)], y=cross_val_accuracies, palette='Blues_r')
plt.title('Accuracy Comparison Across Folds', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.5, 1)
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], 
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.show()

# 3. ROC Curve
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_labels):
    fpr[i], tpr[i], _ = roc_curve(np.array(all_labels) == i, torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, i].numpy())
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']
classes = ['Negative', 'Neutral', 'Positive']

for i in range(num_labels):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC curve for {classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for Sentiment Classification', fontsize=16)
plt.legend(loc='lower right')
plt.show()