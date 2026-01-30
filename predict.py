import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES

def load_trained_model():
    return load_model(MODEL_PATH)

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    prob = prediction[0][0]
    class_name = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    print(f"Prediction: {class_name} (Probability: {prob:.4f})")
    return class_name, prob

def predict_random(df, model):
    img_path = df['Images'].sample(1).iloc[0]
    return predict_image(img_path, model)

if __name__ == "__main__":
    model = load_trained_model()
    # For demo, need df, but since it's script, perhaps load data
    from data_loader import load_data, clean_data
    df = load_data()
    df = clean_data(df)
    predict_random(df, model)