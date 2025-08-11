import tensorflow as tf
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense,
    Resizing,
    Rescaling,
    RandomContrast,
    RandomRotation,
    RandomFlip,
)
from tensorflow.keras.regularizers import L2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Resizing, Rescaling

from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy

import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow_probability as tfp

train_directory = "kaggle/train"
val_directory = "kaggle/test"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "N_EPOCHS": 10,
    "LEARNING_RATE": 0.001,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 100,
    "N_DENSE_2": 10,
    "NUM_CLASSES": len(CLASS_NAMES),
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

# Data Augmentation


augment_layers = tf.keras.Sequential(
    [
        RandomRotation(factor=(-0.025, 0.025)),
        RandomFlip(mode="horizontal"),
        RandomContrast(factor=0.1),
    ]
)


@tf.function
def augment_layer(image, label):
    return augment_layers(image, training=True), label


# Cutmix Augmentation


def box(lamda):
    r_x = tf.cast(
        tfp.distributions.Uniform(
            0.0, tf.cast(CONFIGURATION["IM_SIZE"] - 1, tf.float32)
        ).sample(),
        dtype=tf.int32,
    )
    r_y = tf.cast(
        tfp.distributions.Uniform(
            0.0, tf.cast(CONFIGURATION["IM_SIZE"] - 1, tf.float32)
        ).sample(),
        dtype=tf.int32,
    )

    r_w = tf.cast(CONFIGURATION["IM_SIZE"] * tf.math.sqrt(1 - lamda), dtype=tf.int32)
    r_h = tf.cast(CONFIGURATION["IM_SIZE"] * tf.math.sqrt(1 - lamda), dtype=tf.int32)

    r_x = tf.cast(
        tf.clip_by_value(
            tf.cast(r_x, tf.float32) - tf.cast(r_w, tf.float32) / 2,
            0,
            CONFIGURATION["IM_SIZE"],
        ),
        tf.int32,
    )
    r_y = tf.cast(
        tf.clip_by_value(
            tf.cast(r_y, tf.float32) - tf.cast(r_h, tf.float32) / 2,
            0,
            CONFIGURATION["IM_SIZE"],
        ),
        tf.int32,
    )

    x_b_r = tf.cast(
        tf.clip_by_value(
            tf.cast(r_x, tf.float32) + tf.cast(r_w, tf.float32) / 2,
            0,
            CONFIGURATION["IM_SIZE"],
        ),
        tf.int32,
    )
    y_b_r = tf.cast(
        tf.clip_by_value(
            tf.cast(r_y, tf.float32) + tf.cast(r_h, tf.float32) / 2,
            0,
            CONFIGURATION["IM_SIZE"],
        ),
        tf.int32,
    )

    r_w_final = x_b_r - r_x
    r_w_final = tf.cond(
        tf.equal(r_w_final, 0),
        lambda: tf.constant(1, dtype=tf.int32),
        lambda: r_w_final,
    )

    r_h_final = y_b_r - r_y
    r_h_final = tf.cond(
        tf.equal(r_h_final, 0),
        lambda: tf.constant(1, dtype=tf.int32),
        lambda: r_h_final,
    )

    return r_y, r_x, r_h_final, r_w_final


def cutmix(train_dataset_1, train_dataset_2):
    (image_1, label_1), (image_2, label_2) = train_dataset_1, train_dataset_2

    lamda = tfp.distributions.Beta(0.2, 0.2).sample()

    r_y, r_x, r_h, r_w = box(lamda)
    crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
    pad_2 = tf.image.pad_to_bounding_box(
        crop_2, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]
    )

    crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(
        crop_1, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]
    )

    image = image_1 - pad_1 + pad_2

    lamda_mix = tf.cast(
        1 - (r_h * r_w) / (CONFIGURATION["IM_SIZE"] * CONFIGURATION["IM_SIZE"]),
        dtype=tf.float32,
    )
    mixed_label = lamda_mix * tf.cast(label_1, dtype=tf.float32) + (
        1 - lamda_mix
    ) * tf.cast(label_2, dtype=tf.float32)

    return image, mixed_label


# Dataset Prepration

train_dataset_1 = train_dataset.map(
    augment_layer, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(buffer_size=tf.data.AUTOTUNE)

train_dataset_2 = train_dataset.map(
    augment_layer, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(buffer_size=tf.data.AUTOTUNE)

mixed_dataset = tf.data.Dataset.zip((train_dataset_1, train_dataset_2))

training_dataset = mixed_dataset.map(
    cutmix, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)


validation_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


resize_rescale_layers = tf.keras.Sequential(
    [
        Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
        Rescaling(1.0 / 255),
    ]
)


# Modeling

lenet_model = tf.keras.Sequential(
    [
        InputLayer(shape=(None, None, 3)),
        resize_rescale_layers,
        Conv2D(
            filters=CONFIGURATION["N_FILTERS"],
            kernel_size=CONFIGURATION["KERNEL_SIZE"],
            strides=CONFIGURATION["N_STRIDES"],
            activation="relu",
            kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"]),
        ),
        BatchNormalization(),
        MaxPool2D(
            pool_size=CONFIGURATION["POOL_SIZE"], strides=CONFIGURATION["N_STRIDES"]
        ),
        Dropout(CONFIGURATION["DROPOUT_RATE"]),
        Conv2D(
            filters=CONFIGURATION["N_FILTERS"] * 2 + 4,
            kernel_size=CONFIGURATION["KERNEL_SIZE"],
            strides=CONFIGURATION["N_STRIDES"],
            activation="relu",
            kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"]),
        ),
        BatchNormalization(),
        MaxPool2D(
            pool_size=CONFIGURATION["POOL_SIZE"], strides=CONFIGURATION["N_STRIDES"] * 2
        ),
        Flatten(),
        Dense(
            CONFIGURATION["N_DENSE_1"],
            activation="relu",
            kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"]),
        ),
        BatchNormalization(),
        Dropout(CONFIGURATION["DROPOUT_RATE"]),
        Dense(
            CONFIGURATION["N_DENSE_2"],
            activation="relu",
            kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"]),
        ),
        BatchNormalization(),
        Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax"),
    ]
)

lenet_model.summary()

# Training


loss_function = CategoricalCrossentropy(from_logits=False)
matrics = [
    CategoricalAccuracy(name="accuracy"),
    TopKCategoricalAccuracy(k=2, name="top_k_accuracy"),
]

lenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss=loss_function,
    metrics=matrics,
)

history = lenet_model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=CONFIGURATION["N_EPOCHS"],
    verbose=1,
)

lenet_model.save("lenet_model.h5")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train_loss", "val_loss"])
plt.show()

# from here

lenet_model.evaluate(validation_dataset)


# load the saved model

lenet_model = load_model(
    "lenet_model.h5", custom_objects={"Resizing": Resizing, "Rescaling": Rescaling}
)

test_image = cv2.imread("kaggle/test/fear/PrivateTest_134207.jpg")
im = tf.constant(test_image, dtype=tf.float32)

im = tf.expand_dims(im, axis=0)

print(CLASS_NAMES[tf.argmax(lenet_model(im), axis=-1).numpy()[0]])


# Confusion Matrix

pridicted = []
labels = []

for im, label in validation_dataset:
    pridicted.append(lenet_model(im))
    labels.append(label.numpy())

    print(np.argmax(labels[:-1], axis=-1).flatten())
    print(np.argmax(pridicted[:-1], axis=-1).flatten())

pred = np.argmax(pridicted[:-1], axis=-1).flatten()
lab = np.argmax(labels[:-1], axis=-1).flatten()

cm = confusion_matrix(lab, pred, labels=np.arange(len(CLASS_NAMES)))
print(cm)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
)
plt.title("Confusion Matrix - {}".format(threshold))
plt.ylabel("Actual")
plt.xlabel("Predicted")
