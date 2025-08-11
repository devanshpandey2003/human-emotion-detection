import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, load_model  # Add load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
ap.add_argument("--model_path", help="model.h5", default=None)  # Add this
mode = ap.parse_args().mode
model_path = ap.parse_args().model_path

# ...existing plot_model_history function...

# Define data generators (only needed for training)
if mode == "train":
    train_dir = "../kaggle/train"
    val_dir = "../kaggle/test"

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

# Load pretrained model or create new one
if model_path and os.path.exists(model_path):
    print(f"Loading pretrained model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully!")
else:
    print("Creating new model...")
    # Create the model
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1))
    )
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))

# If you want to train the model
if mode == "train":
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
    )
    plot_model_history(model_info)
    model.save("model.h5")  # Save full model instead of just weights

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    # If no pretrained model was loaded, try to load weights
    if not (model_path and os.path.exists(model_path)):
        try:
            model.load_weights("model.h5")
            print("Loaded model weights from model.h5")
        except:
            print(
                "No model weights found. Please train the model first or provide a pretrained model."
            )
            exit()

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised",
    }

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    # Fixed: Use built-in cascade or download the file
    facecasc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
            )
            prediction = model.predict(
                cropped_img, verbose=0
            )  # Add verbose=0 to reduce output
            maxindex = int(np.argmax(prediction))
            cv2.putText(
                frame,
                emotion_dict[maxindex],
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            "Video", cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC)
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
