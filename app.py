from flask import Flask, render_template, Response
import cv2, base64, io
from PIL import Image
from tensorflow import keras
from flask_socketio import SocketIO
import tensorflow as tf
import numpy as np
from camera import Camera
from process import webopencv

app = Flask(__name__)
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(webopencv())

#---------------- Video Socket Connections --------------------------#
@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")

def gen_frames():  # generate frame by frame from camera
    #camera = cv2.VideoCapture(-1, cv2.CAP_V4L)
    # We load the xml file
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = keras.models.load_model('maskNet_model10.h5')
    LABELS = ['correct_mask', 'incorrect_mask', 'no_mask']

    while True:
        # Capture frame-by-frame

        img = base64.b64decode(camera.get_frame())
        frame = cv2.cvtColor(np.array(Image.open(io.BytesIO(img))), cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1, 1)  # Flip to act as a mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

        # Draw rectangles around each face
        for (start_x, start_y, width, height) in faces:
            end_x = start_x + width
            end_y = start_y + height
            face = frame[start_y:end_y, start_x:end_x]

            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # face = cv2.resize(face, (64, 64))
            # img_array = tf.keras.preprocessing.image.img_to_array(face)
            # img_array = tf.expand_dims(img_array, 0)  # Create a batch

            resized = cv2.resize(face, (64, 64))
            normalized = resized / 255.0
            img_array = np.reshape(normalized, (1, 64, 64, 3))

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            # print(score)

            # print(
            #     "This image most likely belongs to {} with a {:.2f} percent confidence."
            #     .format(LABELS[np.argmax(score)], 100 * np.max(score))
            # )

            label = f'{LABELS[np.argmax(score)]}, {100 * np.max(score)}'

            if LABELS[np.argmax(score)] == 'incorrect_mask':
                color = (255, 0, 0)
            elif LABELS[np.argmax(score)] == 'correct_mask':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    #socketio.run(app)
    app.run(host='0.0.0.0')

