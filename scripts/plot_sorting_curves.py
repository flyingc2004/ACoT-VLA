import json
from pathlib import Path
import matplotlib.pyplot as plt

metrics_path = Path('/home/xhz/ljz/ACoT-VLA/outputs/sorting_phase_classifier_retrain_20260419/training_metrics.json')
out_path = Path('/home/xhz/ljz/ACoT-VLA/outputs/sorting_phase_classifier_retrain_20260419/training_curves.png')

with metrics_path.open('r', encoding='utf-8') as f:
    data = json.load(f)

hist = data.get('history', [])
if not hist:
    raise SystemExit('No history found in training_metrics.json')

epochs = [d['epoch'] for d in hist]
train_acc = [d['train_acc'] for d in hist]
val_acc = [d['val_acc'] for d in hist]
train_loss = [d['train_loss'] for d in hist]
val_loss = [d['val_loss'] for d in hist]

best_idx = max(range(len(hist)), key=lambda i: hist[i]['val_acc'])
best_epoch = epochs[best_idx]
best_val_acc = val_acc[best_idx]

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)

axes[0].plot(epochs, train_acc, marker='o', linewidth=1.8, label='train_acc')
axes[0].plot(epochs, val_acc, marker='o', linewidth=1.8, label='val_acc')
axes[0].axvline(best_epoch, color='tab:red', linestyle='--', linewidth=1.2, alpha=0.8)
axes[0].set_title(f'Accuracy (best val={best_val_acc:.4f} @ epoch {best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.0, 1.0)
axes[0].legend()

axes[1].plot(epochs, train_loss, marker='o', linewidth=1.8, label='train_loss')
axes[1].plot(epochs, val_loss, marker='o', linewidth=1.8, label='val_loss')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('CrossEntropyLoss')
axes[1].legend()

fig.suptitle('Sorting Phase Classifier Training Curves', fontsize=12)
fig.tight_layout()
fig.savefig(out_path)
print(f'Saved: {out_path}')
print(f'Best val acc: {best_val_acc:.6f} at epoch {best_epoch}')
