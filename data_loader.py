import os
import pandas as pd
import PIL
from sklearn.model_selection import train_test_split
from config import DATA_DIR, TEST_SIZE, RANDOM_STATE, LABELS

def load_data():
    img_path = []
    label = []
    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            for path in os.listdir(class_dir):
                if path.endswith('.jpg'):
                    img_path.append(os.path.join(DATA_DIR, class_name, path))
                    label.append(LABELS[class_name])
    
    df = pd.DataFrame({'Images': img_path, 'labels': label})
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def clean_data(df):
    # Remove invalid images
    invalid_images = []
    for image in df['Images']:
        try:
            img = PIL.Image.open(image)
        except:
            invalid_images.append(image)
    
    df = df[~df['Images'].isin(invalid_images)]
    return df

def split_data(df):
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return train, test