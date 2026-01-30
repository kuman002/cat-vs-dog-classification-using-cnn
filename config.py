import os

# Paths
DATA_DIR = 'data/PetImages'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 128
EPOCHS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Classes
CLASS_NAMES = ['Cat', 'Dog']
LABELS = {'Cat': 0, 'Dog': 1}