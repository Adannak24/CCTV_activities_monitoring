from b import CameraTamperingDetector
from c import FaceDetector
from movement_detection import MotionDetection
from d import CrowdDetection
from flask import Flask, Response, request, redirect, url_for, render_template_string, jsonify
import cv2
import os
import logging
from flask_cors import CORS
import face_recognition


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
FORMAT = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s]: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, filename="logs/logs.out")
logger = logging.getLogger(__name__)

# Global variables to store the status
tampering = "false"
flash = "false"
name = "UNKNOWN"
motion = "false"
crowd = "false"
upload = False

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize trackers (Replace with your actual implementation)
camera_tampering_detector = CameraTamperingDetector()
face_detector = FaceDetector("Encodings/encodings.pkl")
# Create an instance of the MotionDetection class
motion_detector = MotionDetection(threshold=50000)
crowd_detector = CrowdDetection(distance_threshold=50)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_frames_motion(video_path):
    global tampering, flash, name  # Use global variables to store tampering and flash status
    cap_motion = cv2.VideoCapture(video_path)

    while True:
        success, m_frame = cap_motion.read()
        if not success:
            break

        # Use the detect method of the class
        m_frame = motion_detector.detect(m_frame)

        ret, buffer = cv2.imencode('.jpg', m_frame)
        if not success:
            continue
        m_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + m_frame + b'\r\n')


def generate_frames(video_path):
    global tampering, flash, name, motion, crowd  # Use global variables to store tampering and flash status
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Use the detect method of the class
        frame, motion = motion_detector.detect(frame)
        # motion = result

        frame, crowd = crowd_detector.process_frame(frame)

        # Process frame with camera tampering detection
        frame, tampering, flash = camera_tampering_detector.detect_tampering(frame)

        # # Update global variables
        # tampering = "red" if t else "green"
        # flash = "red" if f else "green"

        # Process frame with face detection
        frame, name = face_detector.detect_faces(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/')
# def index():
#     html_content = '''
#         <!DOCTYPE html>
#         <html lang="en">
#         <head>
#             <meta charset="UTF-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#             <title>Live Video Feed</title>
#             <style>
#                 body {
#                     font-family: Arial, sans-serif;
#                     margin: 0;
#                     padding: 0;
#                 }
#                 .container {
#                     display: flex;
#                     flex-direction: row;
#                     align-items: flex-start;
#                     justify-content: center;
#                     height: 100vh;
#                     padding: 20px;
#                 }
#                 .video-container {
#                     flex: 1;
#                     display: flex;
#                     justify-content: center;
#                     align-items: center;
#                     margin-right: 20px;
#                 }
#                 .text-container {
#                     flex: 1;
#                     max-width: 50%;
#                 }
#                 img {
#                     width: 100%;
#                     height: auto;
#                 }
#                 form {
#                     margin-top: 20px;
#                 }
#                 .status-btn {
#                     width: 150px;
#                     height: 50px;
#                     border: none;
#                     color: white;
#                     font-size: 18px;
#                     font-weight: bold;
#                     margin: 10px;
#                     cursor: pointer;
#                     border-radius: 5px;
#                 }
#             </style>
#         </head>
#         <body>
#             <div class="container">
#                 <div class="video-container">
#                     <img src="/video_feed" alt="Live Video Feed">
#                 </div>
#                 <div class="text-container">
#                     <h1>Live Video Feed</h1>
#                     <button id="tampering-btn" class="status-btn">Tampering</button>
#                     <button id="flash-btn" class="status-btn">Flash</button>
#                     <button id="motion-btn" class="status-btn">Motion</button>
#                     <p id="name" style="color: #4CAF50;">NAME: {{ name }}</p>
#                     <p id="crowd" style="color: #4CAF50;">CROWD: {{ crowd }}</p>
#                     <h2>Upload Video</h2>
#                     <form action="/upload" method="post" enctype="multipart/form-data">
#                         <input type="file" name="video" accept=".mp4,.avi,.mov">
#                         <input type="submit" value="Upload">
#                     </form>
#                 </div>
#             </div>
#
#             <script>
#     function updateStatuses() {
#         fetch('/status')
#             .then(response => response.json())
#             .then(data => {
#                 // Update Tampering Button Color
#                 const tamperingBtn = document.getElementById('tampering-btn');
#                 tamperingBtn.style.backgroundColor = data.tampering ? 'red' : 'green';
#
#                 // Update Flash Button Color based on condition
#                 const flashBtn = document.getElementById('flash-btn');
#                 flashBtn.style.backgroundColor = data.flash ? 'red' : 'green';
#
#                 // Update Motion Button Color based on condition
#                 const motionBtn = document.getElementById('motion-btn');
#                 motionBtn.style.backgroundColor = data.motion ? 'red' : 'green';
#
#                 // Update the text for person name and crowd status
#                 document.getElementById('name').textContent = 'PERSON NAME: ' + data.name;
#                 document.getElementById('crowd').textContent = 'CROWD: ' + (data.crowd ? 'YES' : 'NO');
#             });
#     }
#
#     // Update the statuses every 2 seconds
#     setInterval(updateStatuses, 2000);  // Changed from 200 ms to 2000 ms for every 2 seconds
# </script>
#
#         </body>
#         </html>
#         '''
#
#     html_content_upload = '''
#             <!DOCTYPE html>
#             <html lang="en">
#             <head>
#                 <meta charset="UTF-8">
#                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
#                 <title>Live Video Feed</title>
#                 <style>
#                     body {
#                         font-family: Arial, sans-serif;
#                         margin: 0;
#                         padding: 0;
#                     }
#                     .container {
#                         display: flex;
#                         flex-direction: row;
#                         align-items: flex-start;
#                         justify-content: center;
#                         height: 100vh;
#                         padding: 20px;
#                     }
#                     .video-container {
#                         flex: 1;
#                         display: flex;
#                         justify-content: center;
#                         align-items: center;
#                         margin-right: 20px;
#                     }
#                     .text-container {
#                         flex: 1;
#                         max-width: 50%;
#                     }
#                     img {
#                         width: 100%;
#                         height: auto;
#                     }
#                     form {
#                         margin-top: 20px;
#                     }
#                     .status-btn {
#                         width: 150px;
#                         height: 50px;
#                         border: none;
#                         color: white;
#                         font-size: 18px;
#                         font-weight: bold;
#                         margin: 10px;
#                         cursor: pointer;
#                         border-radius: 5px;
#                     }
#                 </style>
#             </head>
#             <body>
#                 <div class="container">
#                     <div class="video-container">
#                         <img src="/video_feed_uploaded" alt="Live Video Feed">
#                     </div>
#                     <div class="text-container">
#                         <h1>Live Video Feed</h1>
#                         <button id="tampering-btn" class="status-btn">Tampering</button>
#                         <button id="flash-btn" class="status-btn">Flash</button>
#                         <button id="motion-btn" class="status-btn">Motion</button>
#                         <p id="name" style="color: #4CAF50;">NAME: {{ name }}</p>
#                         <p id="crowd" style="color: #4CAF50;">CROWD: {{ crowd }}</p>
#                         <h2>Upload Video</h2>
#                         <form action="/upload" method="post" enctype="multipart/form-data">
#                             <input type="file" name="video" accept=".mp4,.avi,.mov">
#                             <input type="submit" value="Upload">
#                         </form>
#                     </div>
#                 </div>
#
#                 <script>
#         function updateStatuses() {
#             fetch('/status')
#                 .then(response => response.json())
#                 .then(data => {
#                     // Update Tampering Button Color
#                     const tamperingBtn = document.getElementById('tampering-btn');
#                     tamperingBtn.style.backgroundColor = data.tampering ? 'red' : 'green';
#
#                     // Update Flash Button Color based on condition
#                     const flashBtn = document.getElementById('flash-btn');
#                     flashBtn.style.backgroundColor = data.flash ? 'red' : 'green';
#
#                     // Update Motion Button Color based on condition
#                     const motionBtn = document.getElementById('motion-btn');
#                     motionBtn.style.backgroundColor = data.motion ? 'red' : 'green';
#
#                     // Update the text for person name and crowd status
#                     document.getElementById('name').textContent = 'PERSON NAME: ' + data.name;
#                     document.getElementById('crowd').textContent = 'CROWD: ' + (data.crowd ? 'YES' : 'NO');
#                 });
#         }
#
#         // Update the statuses every 2 seconds
#         setInterval(updateStatuses, 2000);  // Changed from 200 ms to 2000 ms for every 2 seconds
#     </script>
#
#             </body>
#             </html>
#             '''
#     if upload:
#         return render_template_string(html_content_upload)
#     else:
#         return render_template_string(html_content)


@app.route('/status')
def status():
    global tampering, flash, name, motion, crowd
    # print(tampering, flash, crowd)
    # Convert statuses to "YES" or "NO"
    status_dict = {
        'tampering': "YES" if tampering == True else "NO",
        'flash': "YES" if flash == True else "NO",
        'name': name,
        'motion': "YES" if motion == True else "NO",
        'crowd': "YES" if crowd == True else "NO"
    }
    # Return the current tampering and flash status as JSON
    return jsonify(status_dict)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_motion_detection')
def video_feed_motion_detection():
    return Response(generate_frames_motion(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
        file.save(filename)
        # return redirect(url_for('video_feed_uploaded'))
        return jsonify({"error_code": "0", 'status': "video uploaded successfully"})


@app.route('/video_feed_uploaded')
def video_feed_uploaded():
    global upload
    upload = True
    return Response(generate_frames(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/register_face", methods=['POST'])
def register_face():
    image = request.files['face_image']
    face_name = request.form.get('name')
    image_path = os.path.join("images", f"{face_name}.jpg")
    image.save(image_path)
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Get encoding
    try:
        face_detector.register_face(image_path)
        return jsonify({"error_code": "0", 'status': "face registered successfully!"})
    except:
        os.remove(image_path)
        return jsonify({"error_code": "-1", 'status': "Error occured while extracting face."})

#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
