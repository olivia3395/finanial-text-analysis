import matplotlib.pyplot as plt

# Cross-Validation Accuracy and Loss Visualization
plt.figure(figsize=(14, 6))

# Subplot 1: Cross-validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, kf.n_splits + 1), cross_val_accuracies, marker='o', color='blue', label='Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Subplot 2: Cross-validation Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, kf.n_splits + 1), cross_val_losses, marker='o', color='red', label='Cross-Validation Loss')
plt.title('Cross-Validation Loss per Fold')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Final layout adjustments and display
plt.suptitle(f"Cross-Validation Results: Avg Accuracy: {average_accuracy:.4f}, Avg Loss: {average_loss:.4f}", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()