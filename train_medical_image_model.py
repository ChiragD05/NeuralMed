import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Config
# -----------------------------
TRAIN_DIR = Path("data/radiography/train")
VAL_DIR = Path("data/radiography/val")
TEST_DIR = Path("data/radiography/test")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "medical_image_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "medical_image_classes.json"
CONF_MATRIX_PATH = MODEL_DIR / "medical_image_confusion_matrix.png"
REPORT_PATH = MODEL_DIR / "medical_image_classification_report.json"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42
EPOCHS_STAGE_1 = 10
EPOCHS_STAGE_2 = 5

# -----------------------------
# Load datasets
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Compute class weights
# -----------------------------
label_counts = Counter()
for _, labels in train_ds.unbatch():
    label_counts[int(labels.numpy())] += 1

class_weight = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(1024)]),
)

class_weight_dict = {i: float(w) for i, w in enumerate(class_weight)}
print("Class weights:", class_weight_dict)

# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)

# -----------------------------
# Model
# -----------------------------
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
)
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
    ),
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

# -----------------------------
# Stage 1: train classifier head
# -----------------------------
history_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE_1,
    callbacks=callbacks,
    class_weight=class_weight_dict,
)

# -----------------------------
# Stage 2: fine-tune
# -----------------------------
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE_2,
    callbacks=callbacks,
    class_weight=class_weight_dict,
)

# Save final model
model.save(MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")

# -----------------------------
# Final evaluation on test set
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# -----------------------------
# Predictions for report
# -----------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model(images, training=False).numpy()
    pred_classes = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy().tolist())
    y_pred.extend(pred_classes.tolist())

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"Saved classification report to: {REPORT_PATH}")

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH, dpi=200)
plt.close()

print(f"Saved confusion matrix to: {CONF_MATRIX_PATH}")