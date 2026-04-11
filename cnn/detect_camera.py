import cv2
import numpy as np
import tensorflow as tf
import os

from fuzzy.fuzzy import fuzzy_rules

MODEL_PATH = 'cnn/cnn_model.h5'
IMG_SIZE = (128, 128)


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Không tìm thấy mô hình đã được lưu trữ")
        exit()
    return tf.keras.models.load_model(MODEL_PATH)


def choose_camera():
    print("Chọn camera:")
    print("1. Camera Laptop")
    print("2. Camera Điện thoại")

    choice = input("Chọn: ")

    if choice == '1':
        return 0, "Camera Laptop"
    else:
        url = input("Nhập URL Camera Điện thoại (vd: rtsp://): ")
        return url, "Camera Điện thoại"
    
def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_level(score):
    if score < 0.5:
        return "Không có vết bẩn"
    elif score < 0.75:
        return "Vết bẩn nhẹ"
    elif score < 0.9:
        return "Vết bẩn trung bình"
    else:
        return "Vết bẩn nặng"

def detect():
    model = load_model()

    source, name = choose_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Không thể mở Camera:", name)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = preprocess(frame)
        pred = model.predict(img)[0][0]

        if pred >= 0.5:
            label = "Vết bẩn"
            color = (0, 0, 255)
            confidence = pred
        else:
            label = "Không có vết bẩn"
            color = (0, 255, 0)
            confidence = 1 - pred
        
        level = get_level(pred)

        #hiện thi chữ
        cv2.putText(frame, f"Nguồn; {name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Kết quả: {label}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Mức độ: {level}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Độ chính xác: {confidence:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        #Viền
        h, w, _ = frame.shape[:2]
        cv2.rectangle(frame, (5, 5), (w-5, h-5), color, 3)

        cv2.imshow("Nhận diện vết bẩn", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def run_camera():
    detect()

if __name__ == "__main__":
    run_camera()