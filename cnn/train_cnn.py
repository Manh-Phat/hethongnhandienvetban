import tensorflow as tf
from tensorflow.keras import layers, models
import os

DATASET_DIR = "dataset"
MODEL_PATH = "cnn/cnn_model.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32 
EPOCHS = 10

def main():
    if not os.path.exists(DATASET_DIR):
        print("khong tim thay thu muc dataset")
        return
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    model = models.Sequential([
        layers.Rescaling(1./255),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_ds, epochs=EPOCHS)

    model.save(MODEL_PATH)
    print("da luu mo hinh vao:", MODEL_PATH)
if __name__ == "__main__":
    main()
    