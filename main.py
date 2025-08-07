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
)
from tensorflow.keras.regularizers import L2
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy

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

lenet_model = tf.keras.Sequential(
    [
        InputLayer(shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3)),
        Rescaling(1.0 / 255, name="rescaling"),
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
        Dense(CONFIGURATION["NUM_CLASSES"], activation="sofmax"),
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

test_image = cv2.imread("kaggle\test\happy\PrivateTest_218533.jpg")
im = tf.constant(test_image, dtype=tf.float32)

im = tf.expand_dims(im, axis=0)

print(lenet_model(im))


from tensorflow.keras.models import load_model

lenet_model = load_model("lenet_model.h5")
