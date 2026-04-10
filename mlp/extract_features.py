import os
import cv2
import numpy as np

def extract_and_save_features(dataset_path='dataset', save_dir='mlp'):
    print("TRÍCH XUẤT ĐẶC TRƯNG CHO MLP...\n" + "-"*40)
    
    features = []
    labels = []
    label_mapping = {}

    IMG_SIZE = (128, 128) 

    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy thư mục '{dataset_path}'.")
        return

    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    for label_id, folder_name in enumerate(folders):
        label_mapping[label_id] = folder_name
        folder_path = os.path.join(dataset_path, folder_name)
        
        print(f"Đang xử lý thư mục: {folder_name} (Nhãn ID: {label_id})")
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            img = cv2.imread(file_path)
            if img is None:
                continue

            img_resized = cv2.resize(img, IMG_SIZE)

            img_flattened = img_resized.flatten()
            
            features.append(img_flattened)
            labels.append(label_id)

    features = np.array(features)
    labels = np.array(labels)

    import json
    with open(os.path.join(save_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)

    np.save(os.path.join(save_dir, 'features.npy'), features)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)

    print("-" * 40)
    print(f"Đã lưu đặc trưng của {len(features)} ảnh.")
    print(f"Features shape: {features.shape}")
    print(f"Các file đã được lưu tại thư mục '{save_dir}/'")

if __name__ == "__main__":
    extract_and_save_features()