from flask import Flask, render_template, Response, jsonify
import cv2
import frame_setup
from flask_cors import CORS, cross_origin
import model

cam = cv2.VideoCapture(0)
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"/*": {"origins": ["ADDRESS", "ADDRESS"]}})
emotion = ''


def generate_frames():
    global emotion

    CNN, CNN_model = model_runtime()

    while True:
        success, frame = cam.read()
        setup = frame_setup.Frame(CNN, CNN_model)
        frame, emotion = setup.get_frame(frame)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def model_runtime():
    CNN = model.Model()
    CNN_model = CNN.build_model()

    return CNN, CNN_model

# FLASK FUNCTIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/current_emotion')
@cross_origin()
def current_emotion():
    return jsonify(emotion=emotion)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
