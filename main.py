from data_loader import load_data, clean_data, split_data
from evaluate import plot_distribution, plot_sample_images
from train import train_model
from predict import load_trained_model, predict_random
import os

def main():
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    
    # Plot distribution
    n_cats = len(df[df['labels'] == 0])
    n_dogs = len(df[df['labels'] == 1])
    plot_distribution(n_cats, n_dogs)
    
    # Plot sample images
    plot_sample_images(df, 0, 'Cat')
    plot_sample_images(df, 1, 'Dog')
    
    # Train model
    history = train_model()
    
    # Plot history
    from evaluate import plot_history
    plot_history(history)
    
    # Predict
    model = load_trained_model()
    predict_random(df, model)

if __name__ == "__main__":
    main()