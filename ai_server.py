import asyncio
import base64
import cv2
import numpy as np
import os
import time
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =================== Config ===================
CORS_ALLOWED_ORIGINS = ["*"]
HOST = "0.0.0.0"
PORT = 8001

# Smoothing (giữ giống SVM server cũ)
CONFIDENCE_THRESHOLD = 0.35
PREDICTION_HISTORY_SIZE = 5
STABILITY_THRESHOLD = 3

# Model paths
ARCH_FROM_H5 = os.path.join('Structure','Structure 6 (final)','Structure6.h5')
WEIGHTS_PATH = os.path.join('Structure','Structure 0','Structure0.h5')
WINDOW_LEN = 60
HAND_ORDER = 'lh_rh'   # hoặc 'rh_lh' nếu cần

# =================== Labels (hard-coded) ===================
ACTIONS = [
    "ban dang lam gi", "ban di dau the", "ban hieu ngon ngu ky hieu khong", "ban hoc lop may", "ban khoe khong",
    "ban muon gio roi", "ban phai canh giac", "ban ten la gi", "ban tien bo day", "ban trong cau co the",
    "bo me toi cung la nguoi Diec", "cai nay bao nhieu tien", "cai nay la cai gi", "cam on", "cap cuu", "chuc mung",
    "chung toi giao tiep voi nhau bang ngon ngu ky hieu", "con yeu me", "cong viec cua ban la gi", "hen gap lai cac ban",
    "mon nay khong ngon", "toi bi chong mat", "toi bi cuop", "toi bi dau dau", "toi bi dau hong", "toi bi ket xe",
    "toi bi lac", "toi bi phan biet doi xu", "toi cam thay rat hoi hop", "toi cam thay rat vui", "toi can an sang",
    "toi can di ve sinh", "toi can gap bac si", "toi can phien dich", "toi can thuoc", "toi dang an sang",
    "toi dang buon", "toi dang o ben xe", "toi dang o cong vien", "toi dang phai cach ly", "toi dang phan van",
    "toi di sieu thi", "toi di toi Ha Noi", "toi doc kem", "toi khoi benh roi", "toi khong dem theo tien",
    "toi khong hieu", "toi khong quan tam", "toi la hoc sinh", "toi la nguoi Diec", "toi la tho theu",
    "toi lam viec o cua hang", "toi nham dia chi", "toi song o Ha Noi", "toi thay doi bung", "toi thay nho ban",
    "toi thich an mi", "toi thich phim truyen", "toi viet kem", "xin chao"
]

# =================== FastAPI app ===================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================== MediaPipe ===================
mp_hands = mp.solutions.hands
def create_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def hand_to_np(hand):
    if hand is None:
        return np.zeros(21*3, dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32).flatten()

def extract_keypoints_hands(results, hand_order='lh_rh'):
    lh = hand_to_np(results.left_hand_landmarks if results else None)
    rh = hand_to_np(results.right_hand_landmarks if results else None)
    if hand_order == 'rh_lh':
        return np.concatenate([rh, lh], axis=0)
    return np.concatenate([lh, rh], axis=0)

def mp_process_bgr(image_bgr, hands_instance):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_instance.process(image_rgb)
    return results

# =================== Model ===================
def inspect_structure(h5_path):
    import h5py
    info = {}
    with h5py.File(h5_path, 'r') as f:
        def visit(n, o):
            if isinstance(o, h5py.Dataset) and n.endswith('kernel:0'):
                shape = o.shape
                if len(shape) == 2:
                    if '/lstm_2/' in n:
                        info['lstm_2_units'] = int(shape[1] // 4)
                    elif '/lstm_1/' in n:
                        info['lstm_1_units'] = int(shape[1] // 4)
                    elif '/lstm/' in n:
                        info['lstm_units'] = int(shape[1] // 4)
                    if '/dense_1/' in n:
                        info['dense_1_units'] = int(shape[1])
                    elif '/dense/' in n and 'dense_1' not in n and 'dense_2' not in n:
                        info['dense_units'] = int(shape[1])
                    if '/dense_2/' in n:
                        info['num_classes'] = int(shape[1])
        f.visititems(visit)
    info.setdefault('lstm_units', 64)
    info.setdefault('lstm_1_units', 128)
    info.setdefault('lstm_2_units', 64)
    info.setdefault('dense_units', 64)
    info.setdefault('dense_1_units', 32)
    return info

def build_model(info, num_classes, window_len):
    m = Sequential()
    m.add(LSTM(info.get('lstm_units',64), return_sequences=True, activation='relu', input_shape=(window_len,126)))
    m.add(LSTM(info.get('lstm_1_units',128), return_sequences=True, activation='relu'))
    m.add(LSTM(info.get('lstm_2_units',64), return_sequences=False, activation='relu'))
    m.add(Dense(info.get('dense_units',64), activation='relu'))
    m.add(Dense(info.get('dense_1_units',32), activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    return m

# =================== Predictor ===================
class LSTMPredictor:
    def __init__(self, model, labels, window_len=60, hand_order='lh_rh'):
        self.model = model
        self.labels = labels
        self.window_len = window_len
        self.hand_order = hand_order
        self.seq = []
        self.pred_history = []
        self.last_sent = None

    def reset_segment(self):
        self.seq.clear()
        self.pred_history.clear()
        self.last_sent = None

    def process_bgr(self, frame_bgr, hands_instance):
        """Return (label, conf) when stable; '' to clear; or None to send nothing."""
        results = mp_process_bgr(frame_bgr, hands_instance)
        hands_present = bool(results and (results.left_hand_landmarks or results.right_hand_landmarks))

        if not hands_present:
            if self.last_sent is not None:
                self.reset_segment()
                return ""
            self.reset_segment()
            return None

        feat = extract_keypoints_hands(results, self.hand_order)
        self.seq.append(feat)
        if len(self.seq) < self.window_len:
            return None
        self.seq = self.seq[-self.window_len:]

        x = np.array(self.seq, dtype=np.float32)[None, ...]
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else f'class_{idx}'

        if conf >= CONFIDENCE_THRESHOLD:
            self.pred_history.append(label)
            if len(self.pred_history) > PREDICTION_HISTORY_SIZE:
                self.pred_history.pop(0)
            if self.pred_history.count(label) >= STABILITY_THRESHOLD:
                if label != self.last_sent:
                    self.last_sent = label
                    return (label, conf)
        return None

# =================== Initialize ===================
try:
    labels = ACTIONS[:]  # dùng danh sách hard-coded
    info = inspect_structure(ARCH_FROM_H5)
    model = build_model(info, num_classes=len(labels), window_len=WINDOW_LEN)
    model.load_weights(WEIGHTS_PATH)
except Exception as e:
    print(f"[FATAL] Failed to load LSTM model/labels: {e}")
    labels = []
    model = None

# =================== WebSocket ===================
@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if model is None or not labels:
        await websocket.send_text("Server error: model not loaded")
        await websocket.close()
        return

    hands_instance = create_hands()
    predictor = LSTMPredictor(model, labels, window_len=WINDOW_LEN, hand_order=HAND_ORDER)

    MIN_INTERVAL = 0.15
    last_infer = 0.0

    try:
        while True:
            data = await websocket.receive_text()
            if not data.startswith('data:image/jpeg;base64,'):
                continue
            base64_data = data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            now = time.time()
            if now - last_infer < MIN_INTERVAL:
                continue
            last_infer = now

            result = predictor.process_bgr(img_np, hands_instance)
            if result is None:
                continue
            if result == "":
                await websocket.send_text("")   # clear
                continue
            label, conf = result
            await websocket.send_text(f"{label} ({int(conf*100)}%)")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        hands_instance.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
