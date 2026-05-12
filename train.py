"""
train.py  –  OCTDL Eye Disease Classifier
Target: >83% val accuracy

Two-phase approach that is proven to work:
  Phase 1 – Frozen MobileNetV2 base, train head only (reaches ~80%)
  Phase 2 – Unfreeze last 30 layers of base, very low LR (pushes higher)

No oversampling, no label smoothing, sqrt class weights.

Usage:
    python train.py                  # full train
    python train.py --evaluate-only  # evaluate saved model
    python train.py --predict <img>  # predict single image
"""

import os, sys, argparse, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join("OCTDL", "OCTDL")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
VAL_SPLIT    = 0.15
MODEL_PATH   = "model.h5"
CLASSES_JSON = "class_indices.json"
PLOTS_DIR    = "plots"

CLASSES     = ["AMD", "DME", "ERM", "NO", "RAO", "RVO", "VID"]
NUM_CLASSES = len(CLASSES)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def count_images():
    return {
        cls: len([f for f in os.listdir(os.path.join(DATASET_PATH, cls))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if os.path.isdir(os.path.join(DATASET_PATH, cls)) else 0
        for cls in CLASSES
    }


def class_weights(counts):
    total = sum(counts.values())
    raw = {i: total / (NUM_CLASSES * n) if n > 0 else 1.0
           for i, (_, n) in enumerate(counts.items())}
    dampened = {i: np.sqrt(w) for i, w in raw.items()}
    mean_w = np.mean(list(dampened.values()))
    return {i: w / mean_w for i, w in dampened.items()}


def make_generators(augment=True):
    aug = dict(rotation_range=20, width_shift_range=0.12,
               height_shift_range=0.12, shear_range=0.08,
               zoom_range=0.12, horizontal_flip=True, vertical_flip=True,
               brightness_range=[0.85, 1.15], fill_mode="reflect") if augment else {}

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VAL_SPLIT, **aug,
    ).flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training",
        shuffle=True, seed=42, classes=CLASSES,
    )
    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VAL_SPLIT,
    ).flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation",
        shuffle=False, seed=42, classes=CLASSES,
    )
    return train_gen, val_gen


def make_callbacks(min_lr=1e-7, lr_patience=3, es_patience=7):
    return [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=es_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=lr_patience, min_lr=min_lr, verbose=1),
    ]


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = False   # start frozen

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=base.input, outputs=out), base


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_history(h1, h2=None):
    acc  = h1.history["accuracy"]  + (h2.history["accuracy"]  if h2 else [])
    vacc = h1.history["val_accuracy"] + (h2.history["val_accuracy"] if h2 else [])
    loss = h1.history["loss"]      + (h2.history["loss"]      if h2 else [])
    vloss= h1.history["val_loss"]  + (h2.history["val_loss"]  if h2 else [])
    ep   = range(1, len(acc) + 1)
    split = len(h1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y1, y2, title in [(axes[0], acc, vacc, "Accuracy"),
                               (axes[1], loss, vloss, "Loss")]:
        ax.plot(ep, y1, label="Train")
        ax.plot(ep, y2, label="Val")
        if h2:
            ax.axvline(split, color="gray", linestyle="--", label="Fine-tune")
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_history.png"), dpi=150)
    plt.close()


def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 60)
    print("  OCTDL Eye Disease Classifier  –  MobileNetV2")
    print("=" * 60)

    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Dataset not found at '{DATASET_PATH}'"); sys.exit(1)

    counts = count_images()
    print("\nClass distribution:")
    for cls, n in counts.items():
        print(f"  {cls:6s}: {n:5d}")

    cw = class_weights(counts)
    print("\nClass weights:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {cw[i]:.3f}")

    with open(CLASSES_JSON, "w") as f:
        json.dump({cls: i for i, cls in enumerate(CLASSES)}, f, indent=2)

    # ── Phase 1: frozen base ──────────────────────────────────────────────────
    print("\n── Phase 1: frozen base, train head only ──")
    model, base = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(line_length=80)

    train_gen, val_gen = make_generators(augment=True)
    h1 = model.fit(
        train_gen, epochs=20, validation_data=val_gen,
        callbacks=make_callbacks(min_lr=1e-6, lr_patience=3, es_patience=7),
        class_weight=cw, verbose=1,
    )
    best_p1 = max(h1.history["val_accuracy"])
    print(f"\n[Phase 1] Best val_accuracy: {best_p1*100:.2f}%")

    # ── Phase 2: unfreeze top 30 base layers ─────────────────────────────────
    print("\n── Phase 2: unfreeze top 30 base layers, fine-tune ──")

    # Load the best Phase 1 weights
    model = tf.keras.models.load_model(MODEL_PATH)

    # Find MobileNetV2 sub-model by name
    base_layer = next(
        l for l in model.layers if isinstance(l, tf.keras.Model)
    )
    base_layer.trainable = True

    # Freeze all but the last 30 layers of the base
    for layer in base_layer.layers[:-30]:
        layer.trainable = False

    # Keep ALL BatchNorm frozen — critical for stable fine-tuning
    for layer in base_layer.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Trainable params: {trainable_count:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_gen2, val_gen2 = make_generators(augment=True)
    h2 = model.fit(
        train_gen2, epochs=20, validation_data=val_gen2,
        callbacks=make_callbacks(min_lr=1e-8, lr_patience=2, es_patience=6),
        class_weight=cw, verbose=1,
    )
    best_p2 = max(h2.history["val_accuracy"])
    print(f"\n[Phase 2] Best val_accuracy: {best_p2*100:.2f}%")
    print(f"[Overall] Best val_accuracy: {max(best_p1, best_p2)*100:.2f}%")

    plot_history(h1, h2)
    print("\n── Final Evaluation ──")
    evaluate()
    print(f"\n[train] Done. Best model → '{MODEL_PATH}'")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(tta_steps=5):
    if not os.path.exists(MODEL_PATH):
        print(f"[evaluate] '{MODEL_PATH}' not found."); return

    print(f"[evaluate] Loading '{MODEL_PATH}' …")
    model = tf.keras.models.load_model(MODEL_PATH)

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input, validation_split=VAL_SPLIT,
    ).flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation",
        shuffle=False, seed=42, classes=CLASSES,
    )

    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"\n  Val Loss    : {loss:.4f}")
    print(f"  Val Accuracy: {acc*100:.2f}%")

    val_gen.reset()
    y_pred = model.predict(val_gen, verbose=1)
    y_true = val_gen.classes

    if tta_steps > 1:
        print(f"  TTA ({tta_steps} passes) …")
        tta_dg = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10, horizontal_flip=True,
            zoom_range=0.05, validation_split=VAL_SPLIT,
        )
        for _ in range(tta_steps - 1):
            tg = tta_dg.flow_from_directory(
                DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                class_mode="categorical", subset="validation",
                shuffle=False, seed=42, classes=CLASSES,
            )
            y_pred += model.predict(tg, verbose=0)
        y_pred /= tta_steps

    y_cls = np.argmax(y_pred, axis=1)
    print("\n── Classification Report ──")
    print(classification_report(y_true, y_cls, target_names=CLASSES))
    plot_cm(y_true, y_cls)

    print("── Per-class Accuracy ──")
    cm = confusion_matrix(y_true, y_cls)
    for i, cls in enumerate(CLASSES):
        t = cm[i].sum()
        c = cm[i][i]
        bar = "█" * int(c / t * 20) if t > 0 else ""
        print(f"  {cls:6s}: {c/t*100:5.1f}%  ({c:3d}/{t:3d})  {bar}")
    return acc


# ── Single-image prediction ───────────────────────────────────────────────────
def predict_single(image_path, tta_steps=5):
    if not os.path.exists(MODEL_PATH):
        print(f"[predict] '{MODEL_PATH}' not found."); return
    if not os.path.exists(image_path):
        print(f"[predict] Image not found: {image_path}"); return

    model = tf.keras.models.load_model(MODEL_PATH)

    def load_img(path, augment=False):
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        if augment:
            if random.random() > 0.5: arr = arr[:, ::-1, :]
            if random.random() > 0.5: arr = arr[::-1, :, :]
        return np.expand_dims(preprocess_input(arr), 0)

    preds = model.predict(load_img(image_path), verbose=0)
    for _ in range(tta_steps - 1):
        preds += model.predict(load_img(image_path, augment=True), verbose=0)
    preds /= tta_steps

    idx = int(np.argmax(preds[0]))
    print(f"\n── Prediction ──")
    print(f"  Image     : {image_path}")
    print(f"  Predicted : {CLASSES[idx]}  ({preds[0][idx]*100:.2f}%)")
    print(f"\n  All probabilities:")
    for cls, p in zip(CLASSES, preds[0]):
        print(f"    {cls:6s}: {p*100:6.2f}%  {'█' * int(p*40)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", metavar="IMAGE_PATH")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--tta", type=int, default=5)
    args = parser.parse_args()

    if args.predict:
        predict_single(args.predict, tta_steps=args.tta)
    elif args.evaluate_only:
        evaluate(tta_steps=args.tta)
    else:
        train()
