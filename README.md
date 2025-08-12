Human Emotion Detection Using Deep Learning
📌 Introduction
This project detects and classifies human emotions from facial expressions into one of seven categories using Deep Convolutional Neural Networks (CNNs).
It is trained on the FER-2013 dataset published at the International Conference on Machine Learning (ICML).

The dataset contains 35,887 grayscale, 48×48 pixel face images labeled with seven emotions:

Angry 😠

Disgusted 🤢

Fearful 😨

Happy 😀

Neutral 😐

Sad 😢

Surprised 😲

⚙️ Dependencies
Python 3

OpenCV

TensorFlow

Install the required dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
🚀 Basic Usage
1️⃣ Clone the repository and enter the folder:
bash
Copy
Edit
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
2️⃣ Download the FER-2013 dataset
Place the dataset inside the src folder.

▶️ To train the model
bash
Copy
Edit
cd src
python emotions.py --mode train
👀 To run predictions with a pre-trained model
Download the pre-trained model (model1.h5) and place it in the src/webapp folder. Then run:

bash
Copy
Edit
cd src
python emotions.py --mode display
📂 Project Structure
css
Copy
Edit
src/
 ├── webapp/
 │    ├── app.py
 │    ├── templates/
 │    ├── model1.h5
 │    ├── haarcascade_frontalface_default.xml
 │    ├── emotions.py
 │    ├── dataset_prepare.py
 │
 ├── requirements.txt


🧠 Algorithm Workflow
Face Detection — Haar cascade is used to detect faces in webcam/video feed.

Preprocessing — The detected face is resized to 48×48 pixels.

Prediction — The CNN outputs softmax scores for the seven emotion classes.

Display — The emotion with the highest probability is shown on screen.
