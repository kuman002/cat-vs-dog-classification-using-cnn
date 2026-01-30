import matplotlib.pyplot as plt
from config import CLASS_NAMES
import os

def plot_distribution(n_cats, n_dogs):
    images = [n_cats, n_dogs]
    plt.pie(images, labels=CLASS_NAMES, colors=['green', 'blue'], autopct='%1.f%%')
    plt.title('Class Distribution')
    plt.show()

def plot_sample_images(df, class_label, title, num_images=25):
    plt.figure(figsize=(25,25))
    temp = df[df['labels'] == class_label]['Images']
    if len(temp) < num_images:
        num_images = len(temp)
    files = temp.sample(num_images)
    for index, file in enumerate(files):
        plt.subplot(5,5, index+1)
        img = plt.imread(file)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'r', label="Validation Accuracy")
    plt.title('Accuracy Graph')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label="Training Loss")
    plt.plot(epochs, val_loss, 'r', label="Validation Loss")
    plt.title('Loss Graph')
    plt.legend()
    plt.show()