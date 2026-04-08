import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = 'cnn/cnn_model.h5'
IMG_SIZE = (128, 128)


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("khong tim thay mo hinh da duoc luu tru")
        exit()
    return tf.keras.models.load_model(MODEL_PATH)


def choose_camera():
    print("chon camera:")
    print("1. camera laptop")
    print("2. camera dien thoai")

    choice = input("chon: ")

    if choice == '1':
        return 0, "camera laptop"
    else:
        url = input("nhap url camera dien thoai (vd: rtsp://): ")
        return url, "camera dien thoai"
    
def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_level(score):
    if score < 0.5:
        return "khong co vet ban"
    elif score < 0.75:
        return "vet ban nhe"
    elif score < 0.9:
        return "vet ban trung binh"
    else:
        return "vet ban nang"

def detect():
    model = load_model()

    source, name = choose_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("khong the mo camera:", name)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = preprocess(frame)
        pred = model.predict(img)[0][0]

        if pred >= 0.5:
            label = "vet ban"
            color = (0, 0, 255)
            confidence = pred
        else:
            label = "khong co vet ban"
            color = (0, 255, 0)
            confidence = 1 - pred
        
        level = get_level(pred)

        #hiện thi chữ
        cv2.putText(frame, f"Nguon; {name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Ket qua: {label}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Muc do: {level}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Do chinh xac: {confidence:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # vien ngoai
        h, w, _ = frame.shape[:2]
        cv2.rectangle(frame, (5, 5), (w-5, h-5), color, 3)

        cv2.imshow("Nhan dien vet ban", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def run_camera():
    detect()

if __name__ == "__main__":
    run_camera()