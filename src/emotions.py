import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display", required=True)
ap.add_argument(
    "--model_path", help="path to pretrained model (e.g., model.h5)", default=None
)
args = ap.parse_args()
mode = args.mode
model_path = args.model_path


def plot_model_history(model_info):
    """
    Plots accuracy and loss curves given the model_info from model.fit().
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(model_info.history["accuracy"])
    ax1.plot(model_info.history["val_accuracy"])
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Validation"], loc="upper left")

    # Loss
    ax2.plot(model_info.history["loss"])
    ax2.plot(model_info.history["val_loss"])
    ax2.set_title("Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Validation"], loc="upper left")

    plt.tight_layout()
    plt.show()


def create_model():
    """Create and return a new CNN model for emotion recognition."""
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

    return model


# Load or create model
model = None
if model_path and os.path.exists(model_path):
    print(f"Loading pretrained model from: {model_path}")
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model instead...")
        model = create_model()
else:
    if mode == "display":
        # Try to load default model files
        default_models = ["model1.h5", "model.h5"]
        model_loaded = False

        for default_model in default_models:
            if os.path.exists(default_model):
                try:
                    print(f"Loading default model: {default_model}")
                    model = load_model(default_model)
                    print("Model loaded successfully!")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading {default_model}: {e}")
                    continue

        if not model_loaded:
            print(
                "No pretrained model found. Please train the model first or provide a valid model path."
            )
            exit(1)
    else:
        print("Creating new model for training...")
        model = create_model()


# Training mode
if mode == "train":
    # Check if training directories exist
    train_dir = "../kaggle/train"
    val_dir = "../kaggle/test"

    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        print("Please ensure the training data is available.")
        exit(1)

    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        print("Please ensure the validation data is available.")
        exit(1)

    # Training parameters
    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
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

    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    print("Starting training...")
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
        verbose=1,
    )

    # Plot training history
    plot_model_history(model_info)

    # Save the model
    model.save("model1.h5")
    print("Model saved as model1.h5")

# Display mode - Real-time emotion detection
elif mode == "display":
    print("Starting emotion detection...")

    # Prevent OpenCL usage and unnecessary logging
    cv2.ocl.setUseOpenCL(False)

    # Dictionary mapping emotion indices to labels
    emotion_dict = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised",
    }

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit(1)

    # Load face cascade classifier
    try:
        facecasc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if facecasc.empty():
            print("Error: Could not load face cascade classifier")
            exit(1)
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        exit(1)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

            # Extract face region
            roi_gray = gray[y : y + h, x : x + w]

            # Preprocess for model
            try:
                cropped_img = cv2.resize(roi_gray, (48, 48))
                cropped_img = np.expand_dims(
                    cropped_img, axis=-1
                )  # Add channel dimension
                cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension
                cropped_img = cropped_img / 255.0  # Normalize

                # Predict emotion
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                confidence = np.max(prediction)

                # Display emotion and confidence
                emotion_text = f"{emotion_dict[maxindex]} ({confidence:.2f})"
                cv2.putText(
                    frame,
                    emotion_text,
                    (x + 20, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

        # Display frame
        try:
            cv2.imshow(
                "Emotion Detection",
                cv2.resize(frame, (1200, 720), interpolation=cv2.INTER_CUBIC),
            )
        except Exception as e:
            print(f"Error displaying frame: {e}")
            break

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Emotion detection stopped")

else:
    print("Invalid mode. Use --mode train or --mode display")
    exit(1)
