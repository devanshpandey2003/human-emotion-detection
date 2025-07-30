import tensorflow as tf
from keras.layers import Resizing, Rescaling
import matplotlib.pyplot as plt

train_directory = "kaggle/train"
val_directory = "kaggle/test"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
}


# Dataset Loading

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=99,
)


val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)

for i in val_dataset.take(1):
    print(i)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i] / 255.0)
        plt.title(CLASS_NAMES[tf.argmax(labels[i])])
        plt.axis("off")


# Dataset Prepration

training_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


resize_rescale_layers = tf.keras.Sequential(
    [
        Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
        Rescaling(1.0 / 255),
    ]
)


# Modeling
