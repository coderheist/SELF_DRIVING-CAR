import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize SocketIO and Flask
sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10

# Global model variable
model = None

# Image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

# Function to load the trained model
def load_trained_model(model_path):
    try:
        # Load the model without compilation
        loaded_model = load_model(model_path, compile=False)
        # Compile the model with desired optimizer and loss
        loaded_model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3), metrics=['mse'])
        print("Model loaded and compiled successfully.")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Telemetry event handler
@sio.on('telemetry')
def telemetry(sid, data):
    global model
    if model is None:
        print("Model is not loaded.")
        return

    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

# Connect event handler
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Function to send control commands
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

# Main entry point
if __name__ == '__main__':
    model_path = 'model/model (17).h5'
    model = load_trained_model(model_path)
    
    # Wrap Flask application with SocketIO
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
    
    # Run the server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
