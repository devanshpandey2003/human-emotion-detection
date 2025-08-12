from flask import Flask, render_template, jsonify, request, send_file
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image, ImageDraw
import os
import sys
import tempfile

app = Flask(__name__)

# Global variables
model = None
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Emotion colors for UI
emotion_colors = {
    "Happy": "#4CAF50",
    "Sad": "#2196F3",
    "Angry": "#F44336",
    "Surprised": "#FF9800",
    "Fearful": "#9C27B0",
    "Disgusted": "#795548",
    "Neutral": "#607D8B",
}

# Emotion emojis
emotion_emojis = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Surprised": "😲",
    "Fearful": "😨",
    "Disgusted": "🤢",
    "Neutral": "😐",
}


def load_emotion_model(model_name="model1.h5"):
    """Load the trained emotion recognition model with detailed debugging"""
    global model

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try different possible paths
    possible_paths = [
        model_name,  # Current working directory
        os.path.join(script_dir, model_name),  # Same directory as script
        os.path.join(os.getcwd(), model_name),  # Current working directory (explicit)
    ]

    print("=" * 50)
    print("MODEL LOADING DEBUG INFO:")
    print(f"Script directory: {script_dir}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    print(f"Looking for model file: {model_name}")
    print("=" * 50)

    for i, path in enumerate(possible_paths):
        print(f"Attempt {i+1}: Trying path: {path}")

        if os.path.exists(path):
            try:
                file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
                print(f"  ✓ File exists! Size: {file_size:.2f} MB")

                if file_size < 1:  # Model should be at least 1MB
                    print(f"  ⚠ Warning: File seems too small ({file_size:.2f} MB)")

                print(f"  → Loading model from: {path}")
                model = load_model(path)
                print(f"  ✓ Model loaded successfully!")

                # Test the model
                test_input = np.random.random((1, 48, 48, 1))
                test_prediction = model.predict(test_input, verbose=0)
                print(f"  ✓ Model test prediction successful: {test_prediction.shape}")

                return True

            except Exception as e:
                print(f"  ✗ Error loading model: {str(e)}")
                continue
        else:
            print(f"  ✗ File not found at: {path}")

    print("=" * 50)
    print("MODEL LOADING FAILED!")
    print("Available files in current directory:")
    try:
        files = os.listdir(".")
        for file in files:
            if file.endswith((".h5", ".hdf5", ".keras")):
                size = os.path.getsize(file) / (1024 * 1024)
                print(f"  - {file} ({size:.2f} MB)")
    except Exception as e:
        print(f"  Error listing files: {e}")

    print("Available files in script directory:")
    try:
        files = os.listdir(script_dir)
        for file in files:
            if file.endswith((".h5", ".hdf5", ".keras")):
                size = os.path.getsize(os.path.join(script_dir, file)) / (1024 * 1024)
                print(f"  - {file} ({size:.2f} MB)")
    except Exception as e:
        print(f"  Error listing script directory files: {e}")

    print("=" * 50)
    return False


def test_face_detection():
    """Test if OpenCV face detection is working"""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if face_cascade.empty():
            return False, "Face cascade classifier failed to load"

        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = face_cascade.detectMultiScale(test_img)
        return True, "Face detection module working"
    except Exception as e:
        return False, f"Face detection error: {e}"


def preprocess_image(image):
    """Preprocess image for emotion prediction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    image = cv2.resize(image, (48, 48))

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Expand dimensions for model input
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    return image


def detect_faces_and_emotions(image, save_debug_image=False):
    """Detect faces and predict emotions with debugging"""
    print(f"Input image shape: {image.shape}")
    print(f"Input image dtype: {image.dtype}")

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("ERROR: Could not load face cascade classifier")
        return []

    # Handle different image formats
    original_image = image.copy()

    # Convert to BGR if needed (PIL gives RGB)
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] == 3:
            # Assume it's RGB from PIL and convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    print(f"Grayscale image shape: {gray.shape}")
    print(f"Grayscale image dtype: {gray.dtype}")
    print(f"Grayscale image range: {gray.min()} - {gray.max()}")

    # Try multiple face detection approaches
    detection_params = [
        {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (30, 30)},
        {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (20, 20)},
        {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (15, 15)},
        {"scaleFactor": 1.2, "minNeighbors": 3, "minSize": (40, 40)},
    ]

    faces = None
    for i, params in enumerate(detection_params):
        print(f"Trying detection parameters {i+1}: {params}")
        faces = face_cascade.detectMultiScale(gray, **params)
        print(f"Found {len(faces)} faces with parameters {i+1}")
        if len(faces) > 0:
            break

    # If still no faces, try with different preprocessing
    if len(faces) == 0:
        print("Trying histogram equalization...")
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
        )
        print(f"Found {len(faces)} faces after histogram equalization")

    results = []
    debug_image = None

    if save_debug_image:
        debug_image = original_image.copy()
        if len(debug_image.shape) == 3:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

    for i, (x, y, w, h) in enumerate(faces):
        print(f"Processing face {i+1}: x={x}, y={y}, w={w}, h={h}")

        # Draw rectangle on debug image
        if save_debug_image and debug_image is not None:
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                debug_image,
                f"Face {i+1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Extract face region
        face_roi = gray[y : y + h, x : x + w]
        print(f"Face ROI shape: {face_roi.shape}")

        # Preprocess for emotion prediction
        processed_face = preprocess_image(face_roi)
        print(f"Processed face shape: {processed_face.shape}")

        # Predict emotion
        if model:
            try:
                prediction = model.predict(processed_face, verbose=0)
                emotion_index = np.argmax(prediction[0])
                emotion = emotion_dict[emotion_index]
                confidence = float(prediction[0][emotion_index])

                print(f"Predicted emotion: {emotion} ({confidence:.3f})")

                results.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "emotion": emotion,
                        "confidence": confidence,
                        "all_predictions": {
                            emotion_dict[j]: float(prediction[0][j]) for j in range(7)
                        },
                    }
                )

                # Add emotion label to debug image
                if save_debug_image and debug_image is not None:
                    cv2.putText(
                        debug_image,
                        f"{emotion} ({confidence:.2f})",
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            except Exception as e:
                print(f"Error predicting emotion for face {i+1}: {e}")
        else:
            print("No model loaded - cannot predict emotions")
            results.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "emotion": "Unknown",
                    "confidence": 0.0,
                    "all_predictions": {},
                }
            )

    # Save debug image if requested
    if save_debug_image and debug_image is not None:
        debug_path = "debug_image.jpg"
        cv2.imwrite(debug_path, debug_image)
        print(f"Debug image saved to {debug_path}")

    return results


@app.route("/")
def index():
    """Serve the main page with modern UI"""
    face_detection_status, face_detection_msg = test_face_detection()
    model_status = model is not None

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Emotion Recognition</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}

            .header {{
                text-align: center;
                margin-bottom: 40px;
                color: white;
            }}

            .header h1 {{
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}

            .header p {{
                font-size: 1.2rem;
                opacity: 0.9;
                margin-bottom: 30px;
            }}

            .status-panel {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}

            .status-card {{
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 25px;
                color: white;
                transition: all 0.3s ease;
            }}

            .status-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}

            .status-card.success {{
                border-left: 5px solid #4CAF50;
            }}

            .status-card.error {{
                border-left: 5px solid #F44336;
            }}

            .status-icon {{
                font-size: 2rem;
                margin-bottom: 15px;
                display: block;
            }}

            .main-panel {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 25px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}

            .video-container {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-bottom: 40px;
                flex-wrap: wrap;
            }}

            video, canvas {{
                border-radius: 20px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                border: 3px solid #e0e0e0;
                transition: all 0.3s ease;
                max-width: 100%;
            }}

            video:hover, canvas:hover {{
                transform: scale(1.02);
                box-shadow: 0 20px 50px rgba(0,0,0,0.15);
            }}

            .controls {{
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}

            .btn {{
                padding: 15px 30px;
                border: none;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                min-width: 160px;
                justify-content: center;
            }}

            .btn-primary {{
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
            }}

            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
            }}

            .btn-secondary {{
                background: linear-gradient(45deg, #2196F3, #1976D2);
                color: white;
                box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
            }}

            .btn-secondary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(33, 150, 243, 0.4);
            }}

            .btn-danger {{
                background: linear-gradient(45deg, #F44336, #D32F2F);
                color: white;
                box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
            }}

            .btn-danger:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);
            }}

            .btn-upload {{
                background: linear-gradient(45deg, #FF9800, #F57C00);
                color: white;
                box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
            }}

            .btn-upload:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(255, 152, 0, 0.4);
            }}

            .btn:disabled {{
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }}

            .results-container {{
                margin-top: 30px;
            }}

            .loading {{
                text-align: center;
                padding: 40px;
                color: #666;
            }}

            .loading i {{
                font-size: 3rem;
                animation: spin 1s linear infinite;
                color: #4CAF50;
            }}

            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}

            .emotion-card {{
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 20px;
                padding: 25px;
                margin: 20px 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                border-left: 6px solid #4CAF50;
            }}

            .emotion-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.15);
            }}

            .emotion-header {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 20px;
            }}

            .emotion-emoji {{
                font-size: 3rem;
                animation: bounce 2s infinite;
            }}

            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
                40% {{ transform: translateY(-10px); }}
                60% {{ transform: translateY(-5px); }}
            }}

            .emotion-info {{
                flex-grow: 1;
            }}

            .emotion-name {{
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 5px;
            }}

            .emotion-confidence {{
                font-size: 1.2rem;
                color: #666;
            }}

            .confidence-bar {{
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
                overflow: hidden;
                margin: 10px 0;
            }}

            .confidence-fill {{
                height: 100%;
                background: linear-gradient(45deg, #4CAF50, #45a049);
                transition: width 1s ease;
                border-radius: 4px;
            }}

            .predictions-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 10px;
                margin-top: 20px;
            }}

            .prediction-item {{
                background: rgba(255,255,255,0.7);
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                font-size: 0.9rem;
            }}

            .no-faces {{
                text-align: center;
                padding: 60px 20px;
                color: #666;
            }}

            .no-faces i {{
                font-size: 4rem;
                color: #ddd;
                margin-bottom: 20px;
            }}

            .troubleshoot {{
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                border-radius: 15px;
                padding: 25px;
                margin-top: 20px;
            }}

            .troubleshoot h4 {{
                color: #d84315;
                margin-bottom: 15px;
                font-size: 1.3rem;
            }}

            .troubleshoot ul {{
                list-style: none;
                padding-left: 0;
            }}

            .troubleshoot li {{
                padding: 8px 0;
                padding-left: 30px;
                position: relative;
            }}

            .troubleshoot li::before {{
                content: "💡";
                position: absolute;
                left: 0;
                top: 8px;
            }}

            .debug-panel {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9rem;
            }}

            .debug-header {{
                font-weight: bold;
                color: #495057;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            @media (max-width: 768px) {{
                .header h1 {{
                    font-size: 2rem;
                }}
                
                .controls {{
                    flex-direction: column;
                    align-items: center;
                }}
                
                .btn {{
                    width: 100%;
                    max-width: 300px;
                }}
                
                video, canvas {{
                    width: 100%;
                    height: auto;
                }}
            }}

            .pulse {{
                animation: pulse 2s infinite;
            }}

            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
                100% {{ opacity: 1; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-brain"></i> AI Emotion Recognition</h1>
                <p>Real-time emotion detection powered by deep learning</p>
                <div class="profile-links" style="margin-top: 18px;">
                    <a href="https://github.com/devanshpandey2003" target="_blank" style="color: #fff; margin: 0 10px; text-decoration: none;">
                        <i class="fab fa-github"></i> GitHub
                    </a>
                    <a href="https://devanshpandey.vercel.app" target="_blank" style="color: #fff; margin: 0 10px; text-decoration: none;">
                        <i class="fas fa-globe"></i> Portfolio
                    </a>
                    <a href="https://www.instagram.com/devansh_aka_dev" target="_blank" style="color: #fff; margin: 0 10px; text-decoration: none;">
                        <i class="fab fa-instagram"></i> Instagram
                    </a>
                    <a href="https://www.linkedin.com/in/devansh-pandey-43a199258" target="_blank" style="color: #fff; margin: 0 10px; text-decoration: none;">
                        <i class="fab fa-linkedin"></i> LinkedIn
                    </a>
                </div>
            </div>
            
            <div class="status-panel">
                <div class="status-card {'success' if face_detection_status else 'error'}">
                    <i class="status-icon fas fa-{'eye' if face_detection_status else 'exclamation-triangle'}"></i>
                    <h3>Face Detection</h3>
                    <p>{face_detection_msg}</p>
                </div>
                
                <div class="status-card {'success' if model_status else 'error'}">
                    <i class="status-icon fas fa-{'brain' if model_status else 'times-circle'}"></i>
                    <h3>AI Model</h3>
                    <p>{'Emotion model loaded successfully' if model_status else 'Model not loaded - check console'}</p>
                </div>
            </div>
            
            <div class="main-panel">
                <div class="video-container">
                    <video id="video" width="640" height="480" autoplay playsinline></video>
                    <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
                </div>
                
                <div class="controls">
                    <button class="btn btn-primary" onclick="startCamera()">
                        <i class="fas fa-video"></i> Start Camera
                    </button>
                    <button class="btn btn-secondary" onclick="captureAndPredict()">
                        <i class="fas fa-camera"></i> Analyze Emotion
                    </button>
                    <button class="btn btn-danger" onclick="stopCamera()">
                        <i class="fas fa-stop"></i> Stop Camera
                    </button>
                    <button class="btn btn-upload" onclick="testWithFile()">
                        <i class="fas fa-upload"></i> Upload Image
                    </button>
                </div>
                
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="processFile()">
                
                <div id="results" class="results-container"></div>
            </div>
        </div>

        <script>
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let ctx = canvas.getContext('2d');
            let stream = null;

            // Emotion colors and emojis
            const emotionColors = {{
                'Happy': '#4CAF50',
                'Sad': '#2196F3',
                'Angry': '#F44336',
                'Surprised': '#FF9800',
                'Fearful': '#9C27B0',
                'Disgusted': '#795548',
                'Neutral': '#607D8B'
            }};

            const emotionEmojis = {{
                'Happy': '😊',
                'Sad': '😢',
                'Angry': '😠',
                'Surprised': '😲',
                'Fearful': '😨',
                'Disgusted': '🤢',
                'Neutral': '😐'
            }};

            async function startCamera() {{
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{ 
                        video: {{ width: 640, height: 480 }} 
                    }});
                    video.srcObject = stream;
                    showMessage('Camera started successfully! 📹', 'success');
                }} catch (err) {{
                    console.error('Error accessing camera:', err);
                    showMessage('Error accessing camera: ' + err.message, 'error');
                }}
            }}

            function stopCamera() {{
                if (stream) {{
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                    stream = null;
                }}
                showMessage('Camera stopped 📷', 'info');
            }}

            async function captureAndPredict() {{
                if (!stream) {{
                    showMessage('Please start the camera first! 🎥', 'warning');
                    return;
                }}
                
                // Draw video frame to canvas
                ctx.drawImage(video, 0, 0, 640, 480);
                
                // Convert canvas to base64
                let imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                // Send to server for prediction
                try {{
                    showLoading();
                    
                    let response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ image: imageData, save_debug: true }})
                    }});
                    
                    let result = await response.json();
                    displayResults(result);
                }} catch (err) {{
                    console.error('Error predicting:', err);
                    showMessage('Error analyzing image: ' + err.message, 'error');
                }}
            }}

            function testWithFile() {{
                document.getElementById('fileInput').click();
            }}

            async function processFile() {{
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (file) {{
                    const reader = new FileReader();
                    reader.onload = async function(e) {{
                        try {{
                            showLoading();
                            
                            let response = await fetch('/predict', {{
                                method: 'POST',
                                headers: {{
                                    'Content-Type': 'application/json',
                                }},
                                body: JSON.stringify({{ image: e.target.result, save_debug: true }})
                            }});
                            
                            let result = await response.json();
                            displayResults(result);
                        }} catch (err) {{
                            console.error('Error processing file:', err);
                            showMessage('Error processing uploaded image: ' + err.message, 'error');
                        }}
                    }};
                    reader.readAsDataURL(file);
                }}
            }}

            function showLoading() {{
                document.getElementById('results').innerHTML = `
                    <div class="loading">
                        <i class="fas fa-cog pulse"></i>
                        <h3>Analyzing emotions...</h3>
                        <p>Please wait while our AI processes the image</p>
                    </div>
                `;
            }}

            function showMessage(message, type) {{
                const colors = {{
                    success: '#4CAF50',
                    error: '#F44336',
                    warning: '#FF9800',
                    info: '#2196F3'
                }};

                const icons = {{
                    success: 'check-circle',
                    error: 'exclamation-circle',
                    warning: 'exclamation-triangle',
                    info: 'info-circle'
                }};

                document.getElementById('results').innerHTML = `
                    <div style="
                        background: linear-gradient(135deg, ${{colors[type]}}15, ${{colors[type]}}05);
                        border-left: 5px solid ${{colors[type]}};
                        border-radius: 15px;
                        padding: 25px;
                        margin: 20px 0;
                        display: flex;
                        align-items: center;
                        gap: 15px;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    ">
                        <i class="fas fa-${{icons[type]}}" style="font-size: 2rem; color: ${{colors[type]}};"></i>
                        <span style="font-size: 1.1rem; font-weight: 500;">${{message}}</span>
                    </div>
                `;
            }}

            function displayResults(result) {{
                let resultsDiv = document.getElementById('results');
                let html = '';
                
                if (result.debug_info) {{
                    html += `
                        <div class="debug-panel">
                            <div class="debug-header">
                                <i class="fas fa-bug"></i>
                                <span>Debug Information</span>
                            </div>
                            <div>Image shape: ${{result.debug_info.image_shape}}</div>
                            <div>Faces detected: ${{result.debug_info.faces_detected}}</div>
                            <div>Model loaded: ${{result.debug_info.model_loaded}}</div>
                        </div>
                    `;
                }}
                
                if (result.success) {{
                    if (result.results && result.results.length > 0) {{
                        html += '<h2 style="text-align: center; margin: 30px 0; color: #333;"><i class="fas fa-smile"></i> Detected Emotions</h2>';
                        
                        result.results.forEach((face, index) => {{
                            const emotion = face.emotion;
                            const confidence = face.confidence;
                            const emoji = emotionEmojis[emotion] || '🤔';
                            const color = emotionColors[emotion] || '#607D8B';
                            
                            html += `
                                <div class="emotion-card" style="border-left-color: ${{color}};">
                                    <div class="emotion-header">
                                        <div class="emotion-emoji">${{emoji}}</div>
                                        <div class="emotion-info">
                                            <div class="emotion-name" style="color: ${{color}};">${{emotion}}</div>
                                            <div class="emotion-confidence">${{(confidence * 100).toFixed(1)}}% confident</div>
                                        </div>
                                        <div style="text-align: right; color: #666;">
                                            <small>Face ${{index + 1}}</small><br>
                                            <small>${{face.width}}×${{face.height}}px</small>
                                        </div>
                                    </div>
                                    
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${{confidence * 100}}%; background: linear-gradient(45deg, ${{color}}, ${{color}}cc);"></div>
                                    </div>
                            `;
                            
                            if (face.all_predictions && Object.keys(face.all_predictions).length > 0) {{
                                html += '<h4 style="margin: 20px 0 10px 0; color: #555;"><i class="fas fa-chart-bar"></i> All Predictions:</h4>';
                                html += '<div class="predictions-grid">';
                                
                                // Sort predictions by confidence
                                const sortedPredictions = Object.entries(face.all_predictions)
                                    .sort(([,a], [,b]) => b - a);
                                
                                sortedPredictions.forEach(([emotionName, conf]) => {{
                                    const emotionEmoji = emotionEmojis[emotionName] || '🤔';
                                    const isTop = emotionName === emotion;
                                    html += `
                                        <div class="prediction-item" style="
                                            ${{isTop ? `background: ${{color}}20; border: 2px solid ${{color}}; font-weight: bold;` : ''}}
                                        ">
                                            <div style="font-size: 1.2rem;">${{emotionEmoji}}</div>
                                            <div style="font-size: 0.8rem; margin: 5px 0;">${{emotionName}}</div>
                                            <div style="font-size: 0.9rem; font-weight: bold;">${{(conf * 100).toFixed(1)}}%</div>
                                        </div>
                                    `;
                                }});
                                html += '</div>';
                            }}
                            html += '</div>';
                        }});
                        
                        html += `
                            <div style="text-align: center; margin: 30px 0;">
                                <a href="/debug_image" target="_blank" class="btn btn-secondary">
                                    <i class="fas fa-image"></i> View Debug Image
                                </a>
                            </div>
                        `;
                    }} else {{
                        html += `
                            <div class="no-faces">
                                <i class="fas fa-user-slash"></i>
                                <h3>No faces detected in the image</h3>
                                <p>Our AI couldn't find any faces to analyze</p>
                                
                                <div class="troubleshoot">
                                    <h4><i class="fas fa-lightbulb"></i> Tips for better detection:</h4>
                                    <ul>
                                        <li>Ensure your face is well-lit and clearly visible</li>
                                        <li>Look directly at the camera</li>
                                        <li>Remove sunglasses, masks, or other face coverings</li>
                                        <li>Try moving closer to the camera (but not too close)</li>
                                        <li>Avoid shadows or backlighting</li>
                                        <li>Make sure the image is not blurry</li>
                                    </ul>
                                </div>
                                
                                <div style="margin-top: 20px;">
                                    <a href="/debug_image" target="_blank" class="btn btn-secondary">
                                        <i class="fas fa-search"></i> View Debug Image
                                    </a>
                                </div>
                            </div>
                        `;
                    }}
                }} else {{
                    html += `
                        <div style="
                            background: linear-gradient(135deg, #ffebee, #ffcdd2);
                            border-left: 5px solid #F44336;
                            border-radius: 15px;
                            padding: 30px;
                            text-align: center;
                        ">
                            <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: #F44336; margin-bottom: 15px;"></i>
                            <h3 style="color: #c62828; margin-bottom: 10px;">Processing Error</h3>
                            <p style="color: #666; font-size: 1.1rem;">${{result.error}}</p>
                        </div>
                    `;
                }}
                
                resultsDiv.innerHTML = html;
                
                // Smooth scroll to results
                resultsDiv.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }}

            // Initialize with a welcome message
            window.addEventListener('load', function() {{
                showMessage('Welcome! Start your camera or upload an image to begin emotion analysis 🚀', 'info');
            }});
        </script>
    </body>
    </html>
    """


@app.route("/predict", methods=["POST"])
def predict_emotion():
    """Predict emotions from uploaded image with detailed debugging"""
    processing_log = []

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image data provided"})

        image_data = data["image"]
        save_debug = data.get("save_debug", False)

        processing_log.append("Received image data")

        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
            processing_log.append("Removed data URL prefix")

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            processing_log.append(f"Decoded image: {image.size}, mode: {image.mode}")

            # Convert PIL image to numpy array
            image_np = np.array(image)
            processing_log.append(
                f"Converted to numpy: {image_np.shape}, dtype: {image_np.dtype}"
            )

        except Exception as e:
            return jsonify(
                {
                    "success": False,
                    "error": f"Failed to decode image: {str(e)}",
                    "debug_info": {"processing_log": processing_log},
                }
            )

        # Detect faces and emotions
        results = detect_faces_and_emotions(image_np, save_debug_image=save_debug)
        processing_log.append(f"Face detection completed: {len(results)} faces found")

        return jsonify(
            {
                "success": True,
                "results": results,
                "debug_info": {
                    "image_shape": list(image_np.shape),
                    "faces_detected": len(results),
                    "processing_log": processing_log,
                    "model_loaded": model is not None,
                },
            }
        )

    except Exception as e:
        processing_log.append(f"Error: {str(e)}")
        print(f"Error in predict_emotion: {str(e)}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "debug_info": {"processing_log": processing_log},
            }
        )


@app.route("/debug_image")
def debug_image():
    """Serve the debug image"""
    try:
        return send_file("debug_image.jpg", mimetype="image/jpeg")
    except:
        return "Debug image not found", 404


@app.route("/health")
def health_check():
    """Enhanced health check"""
    face_detection_status, face_detection_msg = test_face_detection()

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "face_detection_working": face_detection_status,
            "face_detection_message": face_detection_msg,
            "opencv_version": cv2.__version__,
            "current_directory": os.getcwd(),
            "script_directory": os.path.dirname(os.path.abspath(__file__)),
        }
    )


if __name__ == "__main__":
    print("Starting Emotion Recognition Debug Server...")
    print("=" * 50)

    # Test face detection
    face_status, face_msg = test_face_detection()
    print(f"Face detection test: {face_msg}")

    # Load the model with detailed debugging
    model_loaded = load_emotion_model()

    if not model_loaded:
        print("\n" + "=" * 50)
        print("⚠️  MODEL LOADING FAILED!")
        print("The web app will still work for face detection,")
        print("but emotion prediction will not be available.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✅ MODEL LOADED SUCCESSFULLY!")
        print("All features should work correctly.")
        print("=" * 50)

    print(f"\n🚀 Starting server on http://localhost:5000")
    print("📊 Health check available at: http://localhost:5000/health")
    print("🔍 Debug info will be shown on the main page")
    print("\n" + "=" * 50)

    app.run(debug=True, host="0.0.0.0", port=5000)
