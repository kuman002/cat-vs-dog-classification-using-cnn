from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from config import IMG_SIZE

def create_model():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPool2D((2,2)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model