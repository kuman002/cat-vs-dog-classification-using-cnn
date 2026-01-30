from keras.preprocessing.image import ImageDataGenerator
from data_loader import load_data, clean_data, split_data
from model import create_model
from config import IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH
import os

def create_generators(train_df, test_df):
    train_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_generator = ImageDataGenerator(rescale=1./255)

    train_iterator = train_generator.flow_from_dataframe(
        train_df,
        x_col="Images",
        y_col="labels",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_iterator = val_generator.flow_from_dataframe(
        test_df,
        x_col="Images",
        y_col="labels",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_iterator, val_iterator

def train_model():
    df = load_data()
    df = clean_data(df)
    train_df, test_df = split_data(df)
    
    train_iterator, val_iterator = create_generators(train_df, test_df)
    
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_iterator, epochs=EPOCHS, validation_data=val_iterator)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    
    return history

if __name__ == "__main__":
    train_model()