from flask import Flask, render_template, Response
import cv2
import threading
from google.protobuf.json_format import MessageToDict
import numpy as np
import mediapipe as mp
from PIL import Image
import io

app = Flask(__name__)
camera = cv2.VideoCapture(0)
final = "No Hand"


def generate_frames():
    global final
    pred_list = []

    mp_hands = mp.solutions.hands
    frame_no = 0

    e = threading.Event()
    while not e.wait(0.1):
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            # read the camera frame
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            image = Image.open(io.BytesIO(frame))
            img = np.asarray(image)
            image = cv2.flip(img, 1)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)
            if results.multi_handedness != None:
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    handedness_dict = MessageToDict(hand_handedness)
                    if handedness_dict['classification'][0]['score'] > 0.98:
                        pred_list.append(
                            handedness_dict['classification'][0]['label'])
                        frame_no += 1
                    if frame_no >= 8:
                        if pred_list.count('Right') >= 6:
                            final = 'Right'
                        elif pred_list.count('Left') >= 6:
                            final = 'Left'
                        else:
                            final = 'Both'
                        frame_no = 0
                        pred_list = []

            else:
                final = 'No Hand'
                frame_no = 0
                pred_list = []

        print(final)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def hand_prediction_xhr():
    return final


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
