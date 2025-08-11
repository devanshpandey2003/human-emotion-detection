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
    """Serve the main page with enhanced debugging"""
    face_detection_status, face_detection_msg = test_face_detection()
    model_status = model is not None

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Recognition - Debug Version</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .container {{ text-align: center; }}
            .status {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .status.ok {{ background: #e8f5e8; }}
            .status.error {{ background: #ffeaea; }}
            video, canvas {{ border: 2px solid #ddd; border-radius: 10px; margin: 10px; }}
            button {{ padding: 12px 24px; margin: 5px; font-size: 16px; cursor: pointer; 
                      background: #4CAF50; color: white; border: none; border-radius: 5px; }}
            button:disabled {{ background: #cccccc; cursor: not-allowed; }}
            .results {{ margin-top: 20px; text-align: left; }}
            .debug {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; 
                     font-family: monospace; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Emotion Recognition - Debug Version</h1>
            
            <div class="status {'ok' if face_detection_status else 'error'}">
                <strong>Face Detection:</strong> {face_detection_msg}
            </div>
            
            <div class="status {'ok' if model_status else 'error'}">
                <strong>Emotion Model:</strong> {'Loaded successfully' if model_status else 'Not loaded - check console for details'}
            </div>
            
            <video id="video" width="640" height="480" autoplay></video>
            <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
            <br>
            <button onclick="startCamera()">Start Camera</button>
            <button onclick="captureAndPredict()">Capture & Analyze</button>
            <button onclick="stopCamera()">Stop Camera</button>
            <button onclick="testWithFile()">Upload Image</button>
            
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="processFile()">
            
            <div id="results" class="results"></div>
        </div>

        <script>
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let ctx = canvas.getContext('2d');
            let stream = null;

            async function startCamera() {{
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{ 
                        video: {{ width: 640, height: 480 }} 
                    }});
                    video.srcObject = stream;
                    document.getElementById('results').innerHTML = '<div class="status ok">Camera started successfully</div>';
                }} catch (err) {{
                    console.error('Error accessing camera:', err);
                    document.getElementById('results').innerHTML = 
                        '<div class="status error">Error accessing camera: ' + err.message + '</div>';
                }}
            }}

            function stopCamera() {{
                if (stream) {{
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                    stream = null;
                }}
                document.getElementById('results').innerHTML = '<div class="status">Camera stopped</div>';
            }}

            async function captureAndPredict() {{
                if (!stream) {{
                    alert('Please start the camera first');
                    return;
                }}
                
                // Draw video frame to canvas
                ctx.drawImage(video, 0, 0, 640, 480);
                
                // Convert canvas to base64
                let imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                // Send to server for prediction
                try {{
                    document.getElementById('results').innerHTML = '<div class="status">Processing image...</div>';
                    
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
                    document.getElementById('results').innerHTML = 
                        '<div class="status error">Error predicting: ' + err.message + '</div>';
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
                            document.getElementById('results').innerHTML = '<div class="status">Processing uploaded image...</div>';
                            
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
                            document.getElementById('results').innerHTML = 
                                '<div class="status error">Error processing file: ' + err.message + '</div>';
                        }}
                    }};
                    reader.readAsDataURL(file);
                }}
            }}

            function displayResults(result) {{
                let resultsDiv = document.getElementById('results');
                let html = '';
                
                if (result.debug_info) {{
                    html += '<div class="debug"><strong>Debug Info:</strong><br>';
                    html += 'Image shape: ' + result.debug_info.image_shape + '<br>';
                    html += 'Faces detected: ' + result.debug_info.faces_detected + '<br>';
                    html += 'Model loaded: ' + result.debug_info.model_loaded + '<br>';
                    if (result.debug_info.processing_log) {{
                        html += 'Processing log:<br>' + result.debug_info.processing_log.join('<br>');
                    }}
                    html += '</div>';
                }}
                
                if (result.success) {{
                    if (result.results && result.results.length > 0) {{
                        html += '<h3>Detected Emotions:</h3>';
                        result.results.forEach((face, index) => {{
                            html += '<div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0;">';
                            html += '<h4>Face ' + (index + 1) + ': ' + face.emotion + ' (' + (face.confidence * 100).toFixed(1) + '%)</h4>';
                            html += '<p>Position: (' + face.x + ', ' + face.y + ') - ' + face.width + 'x' + face.height + '</p>';
                            
                            if (face.all_predictions) {{
                                html += '<h5>All Predictions:</h5><ul>';
                                for (let [emotion, confidence] of Object.entries(face.all_predictions)) {{
                                    html += '<li>' + emotion + ': ' + (confidence * 100).toFixed(1) + '%</li>';
                                }}
                                html += '</ul>';
                            }}
                            html += '</div>';
                        }});
                        
                        html += '<p><a href="/debug_image" target="_blank">View Debug Image</a></p>';
                    }} else {{
                        html += '<div class="status error">';
                        html += '<h3>No faces detected</h3>';
                        html += '<p><strong>Troubleshooting tips:</strong></p>';
                        html += '<ul>';
                        html += '<li>Ensure your face is well-lit and clearly visible</li>';
                        html += '<li>Look directly at the camera</li>';
                        html += '<li>Move closer to the camera (but not too close)</li>';
                        html += '<li>Remove obstructions like dark glasses or masks</li>';
                        html += '<li>Try different angles or lighting conditions</li>';
                        html += '</ul>';
                        html += '<p><a href="/debug_image" target="_blank">View Debug Image</a> to see what the system detected</p>';
                        html += '</div>';
                    }}
                }} else {{
                    html += '<div class="status error">';
                    html += '<h3>Error:</h3><p>' + result.error + '</p>';
                    html += '</div>';
                }}
                
                resultsDiv.innerHTML = html;
            }}
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
