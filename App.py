import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import cv2

# Define the model
def get_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # 3 output classes for no waste, partial waste, full waste
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and compile the model
model = get_model()

# Generate random synthetic data for training
num_samples = 1000
img_height, img_width = 128, 128
num_classes = 3

# Generate random images as numpy arrays
X_train = np.random.rand(num_samples, img_height, img_width, 3)
y_train = np.random.randint(0, num_classes, size=(num_samples,))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

def preprocess_image(image, target_size=(128, 128)):
    # Resize image to match model input size
    image = cv2.resize(image, target_size)
    # Normalize image
    image = image / 255.0
    # Expand dimensions to match model input
    image = np.expand_dims(image, axis=0)
    return image

def predict_waste_from_camera(model):
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Display the captured frame
        cv2.imshow('Camera', frame)
        
        # Preprocess and predict
        processed_image = preprocess_image(frame)
        prediction = model.predict(processed_image)
        classes = np.argmax(prediction, axis=1)
        
        if classes == 0:
            result = "No Waste"
        elif classes == 1:
            result = "Partial Waste"
        else:
            result = "Full Waste"
        
        print(f'Prediction: {result}')
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Test prediction using camera
predict_waste_from_camera(model)
