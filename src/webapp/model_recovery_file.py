#!/usr/bin/env python3
"""
Model Recovery and Validation Script
This script helps diagnose and fix issues with emotion recognition models
"""

import os
import sys
import numpy as np
import h5py
import json
from pathlib import Path


def analyze_model_file(model_path):
    """Analyze the structure of an H5 model file"""
    print(f"\n{'='*60}")
    print(f"ANALYZING MODEL FILE: {model_path}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"❌ File does not exist: {model_path}")
        return None

    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📁 File size: {file_size:.2f} MB")

    try:
        with h5py.File(model_path, "r") as f:
            print(f"\n🔍 ROOT LEVEL STRUCTURE:")
            print(f"Root keys: {list(f.keys())}")

            # Check for model config
            if "model_config" in f.attrs:
                print(f"✅ Model config found in attributes")
                config = f.attrs["model_config"]
                if isinstance(config, bytes):
                    config = config.decode("utf-8")
                print(f"Config preview: {config[:200]}...")
                return "full_model"

            # Check for Keras-style structure
            keras_keys = ["model_weights", "optimizer_weights"]
            has_keras = any(key in f.keys() for key in keras_keys)

            if has_keras:
                print(f"✅ Keras-style structure detected")
                if "model_weights" in f:
                    print(f"📊 Model weights structure:")
                    explore_group(f["model_weights"], level=1, max_level=3)
                return "keras_format"

            # Check for layer-based structure
            layer_keys = [
                k
                for k in f.keys()
                if "layer" in k.lower() or "dense" in k.lower() or "conv" in k.lower()
            ]
            if layer_keys:
                print(f"✅ Layer-based structure detected")
                print(f"Layer keys: {layer_keys[:10]}...")  # Show first 10
                return "layer_format"

            # Generic exploration
            print(f"🔍 DETAILED STRUCTURE:")
            for key in f.keys():
                explore_group(f[key], key, level=1, max_level=3)

            return "unknown_format"

    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return "error"


def explore_group(group, name="", level=0, max_level=2):
    """Recursively explore HDF5 group structure"""
    indent = "  " * level

    if hasattr(group, "keys"):  # It's a group
        keys = list(group.keys())
        print(f"{indent}📂 {name}: Group with {len(keys)} items")

        if level < max_level and keys:
            for key in keys[:5]:  # Show first 5 items
                try:
                    explore_group(group[key], key, level + 1, max_level)
                except:
                    print(f"{indent}  ❌ {key}: Could not access")
            if len(keys) > 5:
                print(f"{indent}  ... and {len(keys) - 5} more items")
    else:  # It's a dataset
        try:
            shape = group.shape if hasattr(group, "shape") else "unknown"
            dtype = group.dtype if hasattr(group, "dtype") else "unknown"
            print(f"{indent}📄 {name}: Dataset {shape} {dtype}")
        except:
            print(f"{indent}📄 {name}: Dataset (details unavailable)")


def create_model_architecture():
    """Create a standard emotion recognition model architecture"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Conv2D,
            MaxPooling2D,
            Dropout,
            Flatten,
            Dense,
            BatchNormalization,
        )

        print("\n🏗️  CREATING MODEL ARCHITECTURE:")
        print("Building standard CNN for emotion recognition...")

        model = Sequential(
            [
                # First Conv Block
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(48, 48, 1),
                    name="conv2d_1",
                ),
                BatchNormalization(name="batch_normalization_1"),
                Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
                MaxPooling2D(2, 2, name="max_pooling2d_1"),
                Dropout(0.25, name="dropout_1"),
                # Second Conv Block
                Conv2D(128, (3, 3), activation="relu", name="conv2d_3"),
                BatchNormalization(name="batch_normalization_2"),
                Conv2D(128, (3, 3), activation="relu", name="conv2d_4"),
                MaxPooling2D(2, 2, name="max_pooling2d_2"),
                Dropout(0.25, name="dropout_2"),
                # Third Conv Block
                Conv2D(256, (3, 3), activation="relu", name="conv2d_5"),
                BatchNormalization(name="batch_normalization_3"),
                Conv2D(256, (3, 3), activation="relu", name="conv2d_6"),
                MaxPooling2D(2, 2, name="max_pooling2d_3"),
                Dropout(0.25, name="dropout_3"),
                # Dense layers
                Flatten(name="flatten"),
                Dense(1024, activation="relu", name="dense_1"),
                Dropout(0.5, name="dropout_4"),
                Dense(512, activation="relu", name="dense_2"),
                Dropout(0.3, name="dropout_5"),
                Dense(7, activation="softmax", name="predictions"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        print("✅ Model architecture created successfully!")
        print(f"📊 Total parameters: {model.count_params():,}")
        return model

    except ImportError:
        print("❌ TensorFlow not available - cannot create model architecture")
        return None
    except Exception as e:
        print(f"❌ Error creating model architecture: {e}")
        return None


def try_load_model_strategies(model_path):
    """Try different strategies to load the model"""
    print(f"\n🔧 TRYING DIFFERENT LOADING STRATEGIES:")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model, model_from_json
    except ImportError:
        print("❌ TensorFlow not available")
        return None, "tensorflow_not_available"

    strategies = [
        ("Standard load_model", lambda: load_model(model_path)),
        ("Load without compilation", lambda: load_model(model_path, compile=False)),
        ("Load with custom objects", lambda: load_model(model_path, custom_objects={})),
    ]

    for strategy_name, strategy_func in strategies:
        print(f"\n🔄 Trying: {strategy_name}")
        try:
            model = strategy_func()
            print(f"✅ SUCCESS! Model loaded with: {strategy_name}")

            # Test the model
            test_input = np.random.random((1, 48, 48, 1))
            prediction = model.predict(test_input, verbose=0)
            print(f"✅ Model test successful - output shape: {prediction.shape}")

            return model, strategy_name

        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}...")

    # Try loading weights only
    print(f"\n🔄 Trying: Load architecture + weights separately")
    try:
        architecture_model = create_model_architecture()
        if architecture_model:
            architecture_model.load_weights(model_path)
            print(f"✅ SUCCESS! Weights loaded into new architecture")

            # Test the model
            test_input = np.random.random((1, 48, 48, 1))
            prediction = architecture_model.predict(test_input, verbose=0)
            print(f"✅ Model test successful - output shape: {prediction.shape}")

            return architecture_model, "architecture_plus_weights"
    except Exception as e:
        print(f"❌ Failed to load weights: {str(e)[:100]}...")

    return None, "all_strategies_failed"


def save_fixed_model(model, original_path):
    """Save the fixed model in proper format"""
    base_name = Path(original_path).stem

    # Save complete model
    fixed_path = f"{base_name}_fixed.h5"
    try:
        model.save(fixed_path, save_format="h5")
        print(f"✅ Fixed model saved as: {fixed_path}")

        # Verify the saved model
        test_model = tf.keras.models.load_model(fixed_path)
        print(f"✅ Verification: Fixed model loads correctly!")

        return fixed_path
    except Exception as e:
        print(f"❌ Error saving fixed model: {e}")

        # Try saving weights and architecture separately
        try:
            weights_path = f"{base_name}_weights.h5"
            arch_path = f"{base_name}_architecture.json"

            model.save_weights(weights_path)
            with open(arch_path, "w") as f:
                f.write(model.to_json())

            print(f"✅ Saved separately:")
            print(f"   Weights: {weights_path}")
            print(f"   Architecture: {arch_path}")

            return weights_path, arch_path
        except Exception as e2:
            print(f"❌ Error saving separately: {e2}")

    return None


def create_loading_script(model_path, strategy):
    """Create a Python script to load the model"""
    script_content = f'''#!/usr/bin/env python3
"""
Generated model loading script
Strategy used: {strategy}
Original model: {model_path}
"""

import tensorflow as tf
import numpy as np

def load_emotion_model():
    """Load the emotion recognition model"""
    try:
        # Strategy: {strategy}
'''

    if strategy == "architecture_plus_weights":
        script_content += """
        # Create model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Load weights
        model.load_weights('{model_path}')
"""
    else:
        script_content += f"""
        # Load complete model
        model = tf.keras.models.load_model('{model_path}', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"""

    script_content += """
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Test the model loading
if __name__ == "__main__":
    model = load_emotion_model()
    if model:
        # Test prediction
        test_input = np.random.random((1, 48, 48, 1))
        prediction = model.predict(test_input, verbose=0)
        print(f"✅ Test prediction shape: {prediction.shape}")
        print("Model is ready for use!")
    else:
        print("❌ Model loading failed")
"""

    script_path = "load_model.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    print(f"✅ Model loading script created: {script_path}")
    return script_path


def main():
    """Main function to analyze and fix model"""
    print("🚀 EMOTION RECOGNITION MODEL RECOVERY TOOL")
    print("=" * 60)

    # Find model files
    model_files = []
    for ext in ["*.h5", "*.hdf5", "*.keras"]:
        model_files.extend(Path(".").glob(ext))

    if not model_files:
        print("❌ No model files found in current directory")
        print("   Looking for: *.h5, *.hdf5, *.keras files")
        return

    print(f"📁 Found {len(model_files)} model file(s):")
    for i, file in enumerate(model_files):
        print(f"  {i+1}. {file} ({file.stat().st_size / 1024 / 1024:.2f} MB)")

    # Analyze each model file
    for model_file in model_files:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {model_file}")
        print(f"{'='*60}")

        # Analyze structure
        file_type = analyze_model_file(str(model_file))

        if file_type == "error":
            continue

        # Try loading strategies
        model, strategy = try_load_model_strategies(str(model_file))

        if model:
            print(f"\n✅ SUCCESS! Model loaded using: {strategy}")

            # Save fixed version
            print(f"\n💾 SAVING FIXED MODEL:")
            fixed_path = save_fixed_model(model, str(model_file))

            # Create loading script
            print(f"\n📝 CREATING LOADING SCRIPT:")
            script_path = create_loading_script(str(model_file), strategy)

            print(f"\n🎉 MODEL RECOVERY COMPLETE!")
            print(f"   Fixed model: {fixed_path}")
            print(f"   Loading script: {script_path}")

        else:
            print(f"\n❌ Could not recover model: {model_file}")
            print(f"   Reason: {strategy}")

            print(f"\n🔧 MANUAL RECOVERY SUGGESTIONS:")
            print(f"   1. Check if the original training script is available")
            print(f"   2. Try loading with different TensorFlow versions")
            print(f"   3. Contact the model author for the architecture code")
            print(f"   4. Re-train the model with proper saving procedures")


if __name__ == "__main__":
    main()
