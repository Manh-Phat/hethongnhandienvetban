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

def detect():
    model = load_model()

    source, name = choose_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Không thể mở Camera:", name)
        return
    
    class_names = ['mold_stains', 'mud_stains', 'oil_stains', 'yellow_stains']

    label_names_vn = {
        'mold_stains': 'Vết bẩn mốc',
        'mud_stains': 'Vết bẩn bùn đất',
        'oil_stains': 'Vết bẩn dầu mỡ',
        'yellow_stains': 'Vết bẩn ố vàng'
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = preprocess(frame)
        pred = model.predict(img, verbose=0)[0]
        class_id = np.argmax(pred)
        confidence = pred[class_id]

        label = class_names[class_id]
        label_vn = label_names_vn[label]
        level = fuzzy_rules(confidence)

        if level == "Không có vết bẩn":
            color = (0, 255, 0)
        elif level == "Vết bẩn nhẹ":
            color = (0, 255, 255)
        elif level == "Vết bẩn trung bình":
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        #hiện thi chữ
        cv2.putText(frame, f"Nguồn: {name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Kết quả: {label_vn}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Mức độ: {level}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Độ chính xác: {confidence:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        #Viền
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (5, 5), (w-5, h-5), color, 3)

        cv2.imshow("Nhận diện vết bẩn", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def run_camera():
    detect()

if __name__ == "__main__":
    run_camera()