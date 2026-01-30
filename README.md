# Cat vs Dog Image Classifier

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow/Keras.

## Dataset

The dataset consists of images of cats and dogs from the PetImages directory. It includes:
- Cat images in `data/PetImages/Cat/`
- Dog images in `data/PetImages/Dog/`

## Project Structure

- `config.py`: Configuration file with paths and parameters.
- `data_loader.py`: Functions for loading and preprocessing data.
- `model.py`: CNN model definition.
- `train.py`: Training script.
- `evaluate.py`: Evaluation and visualization functions.
- `predict.py`: Inference script for predicting on new images.
- `main.py`: Main script to run the entire pipeline.
- `requirements.txt`: Python dependencies.

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script:
```bash
python train.py
```

Or run the full pipeline:
```bash
python main.py
```

### Making Predictions

Run the prediction script:
```bash
python predict.py
```

This will load a random image from the dataset and predict whether it's a cat or dog.

## Model Architecture

The CNN consists of:
- Conv2D layer with 16 filters
- MaxPooling2D
- Conv2D layer with 32 filters
- MaxPooling2D
- Flatten
- Dense layer with 512 units
- Output Dense layer with 1 unit (sigmoid for binary classification)

## Results

The trained model is saved as `models/model.h5`. Training includes data augmentation and validation.

## Evaluation

The model is evaluated on a test set, and accuracy/loss graphs are plotted during training.