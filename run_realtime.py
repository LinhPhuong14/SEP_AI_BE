#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, cv2, numpy as np, mediapipe as mp, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ====== Model paths ======
ARCH_FROM_H5 = os.path.join('Structure','Structure 6 (final)','Structure6.h5')
WEIGHTS_PATH = os.path.join('Structure','Structure 0','Structure0.h5')
HAND_ORDER = 'lh_rh'  # hoặc 'rh_lh'

# ====== Ẩn: độ dài tensor cho LSTM ======
_TARGET_SEQ_LEN = 60

# ====== Trigger segment ======
NOHAND_DEBOUNCE = 8
MIN_SEGMENT = 20
CONFIDENCE_THRESHOLD = 0.35

# ====== Labels (hard-coded) ======
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

# ====== MediaPipe (Holistic) ======
mp_holistic = mp.solutions.holistic
def create_holistic():
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
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

# ====== Model build ======
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

def build_model(info, num_classes):
    m = Sequential()
    m.add(LSTM(info.get('lstm_units',64), return_sequences=True, activation='relu', input_shape=(_TARGET_SEQ_LEN,126)))
    m.add(LSTM(info.get('lstm_1_units',128), return_sequences=True, activation='relu'))
    m.add(LSTM(info.get('lstm_2_units',64), return_sequences=False, activation='relu'))
    m.add(Dense(info.get('dense_units',64), activation='relu'))
    m.add(Dense(info.get('dense_1_units',32), activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    return m

def _prepare_tensor(segment):
    if len(segment) == 0:
        return None
    if len(segment) >= _TARGET_SEQ_LEN:
        x = np.array(segment[-_TARGET_SEQ_LEN:], dtype=np.float32)
    else:
        last = segment[-1]
        x = np.array(segment + [last] * (_TARGET_SEQ_LEN - len(segment)), dtype=np.float32)
    return x[None, ...]

def main():
    labels = ACTIONS[:]
    info = inspect_structure(ARCH_FROM_H5)
    model = build_model(info, len(labels))
    model.load_weights(WEIGHTS_PATH)

    holistic = create_holistic()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    segment = []
    nohand_count = 0
    sent_this_gap = False
    last_overlay = ''

    try:
        print("[INFO] Segment mode: predict when hands disappear (q to quit)")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera read failed"); break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            hands_present = bool(results and (results.left_hand_landmarks or results.right_hand_landmarks))

            if hands_present:
                nohand_count = 0
                sent_this_gap = False
                feat = extract_keypoints_hands(results, HAND_ORDER)
                segment.append(feat)
            else:
                nohand_count += 1
                if nohand_count == NOHAND_DEBOUNCE:
                    if len(segment) >= MIN_SEGMENT:
                        x = _prepare_tensor(segment)
                        probs = model.predict(x, verbose=0)[0]
                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        if conf >= CONFIDENCE_THRESHOLD:
                            last_overlay = labels[idx] if idx < len(labels) else f'class_{idx}'
                            print(f"[WORD] {last_overlay} ({int(conf*100)}%)")
                            sent_this_gap = True
                        segment = []  # clear sau quyết định
                elif sent_this_gap and nohand_count > NOHAND_DEBOUNCE:
                    # gửi tín hiệu clear UI (ở bản test: chỉ xoá overlay)
                    last_overlay = ''

            cv2.putText(frame, last_overlay, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('realtime_test', frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
