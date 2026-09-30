# path: cifar10_noise_size_experiments.py
"""
Experiment Script:
"To what extent do noise levels (0%,10%,20%,30%) and dataset sizes (100,1000,5000,10000)
affect CNN accuracy on CIFAR-10 classification?"

Outputs:
- experiment_results/results_raw.csv
- experiment_results/results_summary.csv
- heatmaps & line plots
- ANOVA & Cohen’s d effect size
"""

import os
import time
import random
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Stats
from scipy import stats
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATS_MODELS_AVAILABLE = True
except Exception:
    STATS_MODELS_AVAILABLE = False

# -------------------------
# Configurations
# -------------------------
OUTPUT_DIR = "experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_SIZES = [100, 1000, 5000, 10000]
NOISE_LEVELS = [0, 10, 20, 30]
NOISE_TYPES = ['clean', 'gaussian', 'salt_pepper']
REPEATS = 3
BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 3
RANDOM_SEED = 42
TF_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(TF_SEED)

# -------------------------
# Noise Functions
# -------------------------
def add_gaussian_noise(images: np.ndarray, noise_percent: float) -> np.ndarray:
    """Apply Gaussian noise to images."""
    if noise_percent <= 0:
        return images.copy()
    imgs = images.copy().astype('float32')
    std = (noise_percent / 100.0) * 0.5
    noise = np.random.normal(0.0, std, imgs.shape).astype('float32')
    imgs += noise
    imgs = np.clip(imgs, 0.0, 1.0)
    return imgs

def add_salt_and_pepper(images: np.ndarray, noise_percent: float) -> np.ndarray:
    """Apply salt-and-pepper noise to images."""
    if noise_percent <= 0:
        return images.copy()
    imgs = images.copy().astype('float32')
    p = noise_percent / 100.0
    h, w, c = imgs.shape[1], imgs.shape[2], imgs.shape[3]
    num_pixels = int(p * h * w)
    for i in range(imgs.shape[0]):
        coords = np.random.choice(h * w, num_pixels, replace=False)
        rs = coords // w
        cs = coords % w
        for (r, cidx) in zip(rs, cs):
            imgs[i, r, cidx, :] = 0.0 if np.random.rand() < 0.5 else 1.0
    return imgs

# -------------------------
# Balanced Subsampling
# -------------------------
def balanced_sample(X, y, size, seed=None):
    """Sample balanced subset across 10 CIFAR-10 classes."""
    if seed is not None:
        np.random.seed(seed)
    classes = np.unique(y)
    per_class = max(1, size // len(classes))
    indices = []
    for cls in classes:
        cls_idx = np.where(y.flatten() == cls)[0]
        chosen = np.random.choice(cls_idx, per_class, replace=(len(cls_idx) < per_class))
        indices.extend(chosen.tolist())
    indices = np.array(indices)
    if len(indices) < size:
        extra = np.random.choice(np.arange(len(X)), size - len(indices), replace=False)
        indices = np.concatenate([indices, extra])
    np.random.shuffle(indices)
    return X[indices], y[indices]

# -------------------------
# CNN Model (Simple Baseline)
# -------------------------
def build_baseline_cnn(input_shape=(32,32,3), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# -------------------------
# Load CIFAR-10
# -------------------------
(X_train_all, y_train_all), (X_test_all, y_test_all) = cifar10.load_data()
X_train_all = X_train_all.astype('float32') / 255.0
X_test_all = X_test_all.astype('float32') / 255.0
y_train_all = y_train_all.astype('int32')
y_test_all = y_test_all.astype('int32')
y_cat_test = to_categorical(y_test_all, 10)

# Clean test set stays fixed
X_test_fixed, y_cat_test_fixed = X_test_all.copy(), y_cat_test.copy()

# -------------------------
# Experiment Loop
# -------------------------
records = []
total_runs = len(NOISE_TYPES) * len(NOISE_LEVELS) * len(DATASET_SIZES) * REPEATS
pbar = tqdm(total=total_runs, desc="Running experiments")

for noise_type in NOISE_TYPES:
    for noise_level in NOISE_LEVELS:
        for size in DATASET_SIZES:
            for rep in range(1, REPEATS+1):
                start_time = time.time()
                X_sub, y_sub = balanced_sample(X_train_all, y_train_all, size, seed=RANDOM_SEED + rep)

                # Validation split (20%)
                n_val = int(len(X_sub) * 0.2)
                X_val, y_val = X_sub[:n_val], y_sub[:n_val]
                X_train_sub, y_train_sub = X_sub[n_val:], y_sub[n_val:]

                if noise_type == 'clean':
                    X_train_noisy = X_train_sub.copy()
                elif noise_type == 'gaussian':
                    X_train_noisy = add_gaussian_noise(X_train_sub, noise_level)
                elif noise_type == 'salt_pepper':
                    X_train_noisy = add_salt_and_pepper(X_train_sub, noise_level)

                y_cat_train = to_categorical(y_train_sub, 10)
                y_cat_val = to_categorical(y_val, 10)

                datagen = ImageDataGenerator(
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True
                )
                train_gen = datagen.flow(X_train_noisy, y_cat_train, batch_size=BATCH_SIZE, shuffle=True)

                model = build_baseline_cnn()
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

                history = model.fit(
                    train_gen,
                    epochs=EPOCHS,
                    steps_per_epoch=max(1, X_train_noisy.shape[0] // BATCH_SIZE),
                    validation_data=(X_val, y_cat_val),
                    callbacks=[early_stop],
                    verbose=0
                )

                train_time = time.time() - start_time
                test_loss, test_acc = model.evaluate(X_test_fixed, y_cat_test_fixed, verbose=0)

                records.append({
                    'noise_type': noise_type,
                    'noise_level': noise_level,
                    'dataset_size': size,
                    'repeat': rep,
                    'train_time_sec': train_time,
                    'test_accuracy': float(test_acc)
                })
                pbar.update(1)
pbar.close()

# -------------------------
# Save & Analyze
# -------------------------
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "results_raw.csv"), index=False)
summary = df.groupby(['noise_type','noise_level','dataset_size'])['test_accuracy'].agg(['mean','std']).reset_index()
summary.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

print("\n✅ Experiments complete! Results saved in:", OUTPUT_DIR)

# -------------------------
# Visualization
# -------------------------
import seaborn as sns
for nt in NOISE_TYPES:
    sub = summary[summary['noise_type']==nt]
    pivot = sub.pivot(index='noise_level', columns='dataset_size', values='mean')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm")
    plt.title(f"Accuracy Heatmap ({nt})")
    plt.ylabel("Noise Level (%)")
    plt.xlabel("Dataset Size")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{nt}.png"))
    plt.close()

print("📊 Heatmaps saved to experiment_results/")

# -------------------------
# ANOVA (Two-way)
# -------------------------
anova_file = os.path.join(OUTPUT_DIR, "anova_results.txt")
with open(anova_file, "w") as f:
    if STATS_MODELS_AVAILABLE:
        df['noise_level'] = df['noise_level'].astype('category')
        df['dataset_size'] = df['dataset_size'].astype('category')
        model = ols('test_accuracy ~ C(noise_type) + C(noise_level) + C(dataset_size) + \
                    C(noise_type):C(noise_level) + C(noise_type):C(dataset_size) + \
                    C(noise_level):C(dataset_size)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        f.write(str(anova_table))
    else:
        f.write("statsmodels not installed; skipping full ANOVA.")
print("📈 ANOVA results saved.")

print("\n🎯 Done. Check 'experiment_results/' for:")
print("  - results_raw.csv, results_summary.csv")
print("  - heatmap_*.png, anova_results.txt\n")

