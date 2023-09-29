from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import face_recognition
import time

app = Flask(__name__)

path = 'image'
images = []
classNames = []
li = os.listdir(path)
for cls in li:
    current = cv2.imread(f'{path}/{cls}')
    images.append(current)
    classNames.append(os.path.splitext(cls)[0])

def findEncodings(images):
    encodeLi = []
    for i in images:
        img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeLi.append(encode)
    return encodeLi

encodeLiKnown = findEncodings(images)
print('Encoding is done')

cam = cv2.VideoCapture(0)

def generate_frames():
    # global i, fall_counter, fall_threshold,j
    cam = cv2.VideoCapture(0)
    name = ''
    i = 0

    # Initialize fall detection variables
    fitToEllipse = False
    time.sleep(2)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    j = 0
    fall_counter = 0
    fall_threshold = 10
    while True:
        success, img = cam.read()

        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
        faceCur = face_recognition.face_locations(imgSmall)
        encodeCur = face_recognition.face_encodings(imgSmall, faceCur)

        for encodeFace, faceLoc in zip(encodeCur, faceCur):
            match = face_recognition.compare_faces(encodeLiKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeLiKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            print(i)
            i += 1
            if match[matchIndex]:
                name = classNames[matchIndex].upper()
            else:
                name = 'Unknown'

            (y1, x2, y2, x1) = faceLoc
            (y1, x2, y2, x1) = (y1 * 4, x2 * 4, y2 * 4, x1 * 4)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fgmask = fgbg.apply(gray)
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            areas = []

            for contour in contours:
                ar = cv2.contourArea(contour)
                areas.append(ar)

            max_area = max(areas, default=0)
            ma_index = areas.index(max_area)
            cnt = contours[ma_index]

            x, y, w, h = cv2.boundingRect(cnt)

            if h < w:
                j += 1
                if j > fall_threshold:
                    fall_counter += 1
                    print("FALL", fall_counter,name)
                    cv2.putText(img, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                j = 0
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    # Stop the video feed
    cam.release()
    cv2.destroyAllWindows()
    return "Video feed stopped"

if __name__ == "__main__":
    app.run(debug=True)
