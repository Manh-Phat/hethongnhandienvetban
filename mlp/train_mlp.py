import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def train_and_evaluate_mlp(data_dir='mlp'):
    print("HUẤN LUYỆN MÔ HÌNH MLP...\n" + "-"*40)
    
    features_path = os.path.join(data_dir, 'features.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Lỗi: Không tìm thấy file dữ liệu. Vui lòng chạy lại extract_features.py")
        return
        
    X = np.load(features_path)
    y = np.load(labels_path)

    X = X / 255.0

    num_classes = len(np.unique(y))
    
    print(f"Tổng số mẫu ảnh: {X.shape[0]}")
    print(f"Kích thước đầu vào mỗi ảnh (features): {X.shape[1]}")
    print(f"Số loại vết bẩn cần phân loại: {num_classes}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    print("\nĐang huấn luyện mô hình...")
    history = model.fit(X_train, y_train, 
                        epochs=30,
                        batch_size=8, 
                        validation_data=(X_test, y_test))
    
    model_save_path = os.path.join(data_dir, 'mlp_model.keras')
    model.save(model_save_path)
    print(f"\nĐã lưu mô hình tại: {model_save_path}")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("-" * 40)
    print(f"KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Loss: {loss:.4f}")
    
    #Vẽ biểu đồ
    plot_history(history, data_dir)

def plot_history(history, save_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    #Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Test Accuracy')
    plt.title('Biểu đồ Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    #Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Test Loss')
    plt.title('Biểu đồ Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'training_plot.png')
    plt.savefig(plot_path)
    print(f"\nĐã lưu ảnh biểu đồ tại: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    train_and_evaluate_mlp()