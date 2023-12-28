from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Load the pre-trained HOG-based pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Function to capture frames from the camera
def capture_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame to a smaller size for faster processing
        gray = cv2.resize(gray, (640, 480))
        # Detect pedestrians in the grayscale frame
        boxes, weights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
        # Draw bounding boxes around the detected pedestrians
        for (x, y, w, h), confidence in zip(boxes, weights):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f'Confidence: {round(confidence, 2)}'
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        # Yield the frame as a response to the client
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()

