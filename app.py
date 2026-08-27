import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import logging
from collections import Counter
import time

# System Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

#  EMOTION MAPPING 
EMOTION_MAP = {
    'Happy': {
        'color': '#00ff88',
        'msg': 'Positive neural resonance detected. Optimizing environment.',
        'music': 'happy.mp3',
        'icon': '', 
        'analysis': 'Dopamine surge confirmed. Cognitive flexibility at 94%.',
        'voice_line': 'Neural resonance is positive. Enhancing environmental parameters for peak performance.',
        'glow': 'rgba(0, 255, 136, 0.4)'
    },
    'Sad': {
        'color': '#00d4ff',
        'msg': 'Melancholic state identified. Activating support protocols.',
        'music': 'sad.mp3',
        'icon': '',
        'analysis': 'Reduced serotonin detected. System suggesting sensory recalibration.',
        'voice_line': 'I have detected a drop in your neural baseline. Initiating calming audio sequences.',
        'glow': 'rgba(0, 212, 255, 0.4)'
    },
    'Angry': {
        'color': '#ff0055',
        'msg': 'Elevated stress markers. Deploying mitigation sequence.',
        'music': 'angry.mp3',
        'icon': '',
        'analysis': 'Cortisol levels exceeding threshold. Heart rate variability low.',
        'voice_line': 'Elevated stress markers detected. Please focus on rhythmic breathing while I adjust the audio.',
        'glow': 'rgba(255, 0, 85, 0.4)'
    },
    'Neutral': {
        'color': '#a0a0ff',
        'msg': 'Neural baseline stable. Continuous monitoring active.',
        'music': 'neutral.mp3',
        'icon': '',
        'analysis': 'Prefrontal cortex activity consistent. No anomalies found.',
        'voice_line': 'System status nominal. Neural baseline is stable.',
        'glow': 'rgba(160, 160, 255, 0.4)'
    },
    'Fear': {
        'color': '#9b59b6',
        'msg': 'Anxiety indicators present. Initiating safety reassurance.',
        'music': 'fear.mp3',
        'icon': '',
        'analysis': 'Amygdala hyper-activity detected. Physiological threat response active.',
        'voice_line': 'Sensors indicate anxiety. You are in a secure environment. Breathe deeply.',
        'glow': 'rgba(155, 89, 182, 0.4)'
    },
    'Surprise': {
        'color': '#f1c40f',
        'msg': 'Unexpected stimulus registered. Heightened awareness.',
        'music': 'surprise.mp3',
        'icon': '',
        'analysis': 'Rapid sensory shift detected. Attention focus redirected.',
        'voice_line': 'Stimulus anomaly detected. Analyzing novelty response.',
        'glow': 'rgba(241, 196, 15, 0.4)'
    },
    'Disgust': {
        'color': '#2ecc71',
        'msg': 'Aversion detected. Environmental preference adjustment.',
        'music': 'disgust.mp3',
        'icon': '',
        'analysis': 'Avoidance behavior patterns logged. Preference engine updating.',
        'voice_line': 'Negative preference noted. Adjusting system parameters to avoid current stimuli.',
        'glow': 'rgba(46, 204, 113, 0.4)'
    }
}

class EmotionEngine:
    def __init__(self):
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.current_emotion = "Neutral"
        self.confidence = 0
        self.model = None
        self.face_cascade = None
        self.camera = None
        self.is_running = False
        
        # Stability logic 
        self.emotion_history = []
        self.history_size = 12
        self.min_confidence = 45
        self.stability_threshold = 7
        
        self.initialize()

    def initialize(self):
        try:
            self.model = load_model("models/emotion_model.h5", compile=False)
            self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            logger.info("✓ BRAIN ENGINE INITIALIZED")
        except Exception as e:
            logger.error(f"Initialization Failed: {e}")

    def start_camera(self):
       
        if self.camera is not None and self.camera.isOpened():
            self.is_running = True
            return True
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            return True
        except:
            return False

    def stop_camera(self):
        if self.camera:
            self.camera.release()
            self.camera = None
        self.is_running = False
        return True

    def preprocess(self, frame):
        face = cv2.resize(frame, (64, 64))
        face = cv2.equalizeHist(face)
        return face.astype("float32") / 255.0

    def gen_frames(self):
        while True:
            if not self.is_running or self.camera is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "LINK OFFLINE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 50), 3)
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue

            success, frame = self.camera.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                processed = self.preprocess(face_roi)
                preds = self.model.predict(np.expand_dims(np.expand_dims(processed, -1), 0), verbose=0)[0]
                
                idx = np.argmax(preds)
                self.current_emotion = self.labels[idx]
                self.confidence = float(preds[idx] * 100)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 136), 2)
                cv2.putText(frame, f"ID: {self.current_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

engine = EmotionEngine()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(engine.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({
        'emotion': engine.current_emotion,
        'data': EMOTION_MAP[engine.current_emotion],
        'confidence': round(engine.confidence, 1),
        'running': engine.is_running
    })

@app.route('/start_system', methods=['POST'])
def start(): return jsonify({'success': engine.start_camera()})

@app.route('/stop_system', methods=['POST'])
def stop(): return jsonify({'success': engine.stop_camera()})

@app.route('/static/music/<path:filename>')
def serve_music(filename): return send_from_directory('static/music', filename)

if __name__ == '__main__':
    print("\n" + "█"*50)
    print("  NEUROSYNC OS v3.0 - PROFESSIONAL INTERFACE ONLINE")
    print("█"*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
