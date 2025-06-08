#!/usr/bin/env python3
"""
analysis.py

Evaluation protocol:
  • Accuracy
  • Macro-averaged F1
  • Multi-class ROC AUC (One-vs-Rest)
  • RMSE (on predicted probabilities vs. one-hot truth)

Outputs to test/:
  – augmentation_examples.png (original vs. augmented samples)
  – label_distribution.png    (True vs Pred counts)
  – confidence_histogram.png  (distribution of max probabilities)
  – confusion_matrix.png      (per-class confusion)
  – roc_curves.png            (per-class ROC)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    confusion_matrix,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split

# ─── CONFIG ─────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\ASUS\Desktop\ecs111\final project"
MODEL_PATH   = os.path.join(PROJECT_ROOT, "test", "best_model.keras")
CSV_PATH     = os.path.join(PROJECT_ROOT, "training_onehot.csv")
IMG_ROOT     = os.path.join(PROJECT_ROOT, "converted")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
CLASS_NAMES = ['Angry','Fear','Happy','Sad']

# ─── DATA AUGMENTATION PIPELINE (for demo) ─────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.3),
    layers.RandomTranslation(0.1, 0.1),
], name='data_augmentation')

# ─── LOAD & SPLIT CSV ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['fullpath'] = df['imagepath'].apply(
    lambda p: os.path.normpath(os.path.join(IMG_ROOT, p))
)
df = df[df['fullpath'].map(os.path.exists)].reset_index(drop=True)

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label_encoded'],
    random_state=42
)

# ─── PREPARE VALIDATION GENERATOR ─────────────────────────────────
valid_aug = ImageDataGenerator(rescale=1/255.0)
valid_gen = valid_aug.flow_from_dataframe(
    valid_df,
    x_col="fullpath",
    y_col=CLASS_NAMES,
    target_size=IMG_SIZE,
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ─── DEMONSTRATE AUGMENTATION ─────────────────────────────────────
batch_images, _ = next(valid_gen)
augmented_images = data_augmentation(batch_images)

n = min(8, batch_images.shape[0])
plt.figure(figsize=(16, 8))
for i in range(n):
    # original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(batch_images[i])
    plt.axis('off')
    if i == 0:
        ax.set_title('Original')
    # augmented
    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(augmented_images[i].numpy())
    plt.axis('off')
    if i == 0:
        ax.set_title('Augmented')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "augmentation_examples.png"))
plt.close()

# ─── LOAD TRAINED MODEL ───────────────────────────────────────────
print(">> Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)

# ─── EVALUATION ───────────────────────────────────────────────────
steps_val = int(np.ceil(len(valid_df) / BATCH_SIZE))
probs     = model.predict(valid_gen, steps=steps_val, verbose=1)
y_pred    = np.argmax(probs, axis=1)
y_true    = np.argmax(valid_df[CLASS_NAMES].values, axis=1)

acc     = accuracy_score(y_true, y_pred)
f1_mac  = f1_score(y_true, y_pred, average='macro')
roc_auc = roc_auc_score(valid_df[CLASS_NAMES].values, probs,
                        average='macro', multi_class='ovr')
rmse    = np.sqrt(mean_squared_error(valid_df[CLASS_NAMES].values, probs))

print(f"Accuracy   : {acc:.4f}")
print(f"F1 (macro) : {f1_mac:.4f}")
print(f"ROC AUC    : {roc_auc:.4f}")
print(f"RMSE       : {rmse:.4f}")

# ─── VISUALIZATIONS ───────────────────────────────────────────────

# 1) True vs. Predicted Label Distribution (fixed)
true_counts = pd.Series(y_true).value_counts().sort_index()
pred_counts = pd.Series(y_pred).value_counts().sort_index()

true_counts.index = CLASS_NAMES
pred_counts.index = CLASS_NAMES

df_counts = pd.DataFrame({
    'True': true_counts,
    'Pred': pred_counts
})

ax = df_counts.plot.bar(rot=0, figsize=(6, 4))
ax.set_ylabel("Number of samples")
ax.set_title("True vs. Predicted Label Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"))
plt.close()

# 2) Confidence histogram
max_conf = probs.max(axis=1)
plt.figure(figsize=(6, 4))
plt.hist(max_conf, bins=20, range=(0,1), edgecolor='k')
plt.xlabel("Max Predicted Probability")
plt.ylabel("Sample Count")
plt.title("Histogram of Model Confidence")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confidence_histogram.png"))
plt.close()

# 3) Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# 4) Multi-class ROC Curves
plt.figure(figsize=(8, 6))
handles, labels = [], []
for i, cls in enumerate(CLASS_NAMES):
    disp = RocCurveDisplay.from_predictions(
        valid_df[cls].values,
        probs[:, i],
        name=cls,
        ax=plt.gca()
    )
    handles.append(disp.line_)
    auc = roc_auc_score(valid_df[cls].values, probs[:, i])
    labels.append(f"{cls} (AUC={auc:.2f})")

plt.plot([0,1], [0,1], 'k--', linewidth=0.5)
plt.title("Multi-class ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(handles=handles, labels=labels, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
plt.close()

print("All artifacts saved to:", OUTPUT_DIR)
