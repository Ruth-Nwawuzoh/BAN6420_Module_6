# BAN6420_Module_6
Fashion MNIST Classification Using CNN   

## Project Overview
This project is part of the Module 6 Assignment: Fashion MNIST Classification. The objective is to classify images from the Fashion MNIST dataset using a Convolutional Neural Network (CNN) built in both Python and R. The model predicts the class of fashion items (e.g., T-shirts, trousers, etc.) and demonstrates predictions for at least two images from the dataset.

## Dataset
- **Fashion MNIST** dataset: A collection of 70,000 grayscale images (28x28 pixels) across 10 classes.
- Training set: 60,000 images
- Test set: 10,000 images
- Each image is 28x28 pixels, and the corresponding label is an integer between 0 and 9 representing one of the fashion categories. 
- Classes:  
  0: T-shirt/top  
  1: Trouser  
  2: Pullover  
  3: Dress  
  4: Coat  
  5: Sandal  
  6: Shirt  
  7: Sneaker  
  8: Bag  
  9: Ankle boot  

- The dataset is loaded using the `keras.datasets.fashion_mnist` module in Python and the `dataset_fashion_mnist()` function in R.

## Requirements
### Python Dependencies
- Ensure you have Python 3.8+ installed. 
- Install the following libraries by running: numpy, matplotlib, keras, tensorflow

#### R Dependencies
- Ensure R version 4.0+ is installed. Install the required R libraries by running:
- install.packages(c("keras", "tensorflow"))
- For R, ensure you have TensorFlow set up:
  library(tensorflow)
  install_tensorflow()

## Running the Code
### Python
1. Download the zipped file ('Fashion_MNIST_CNN.zip')and unzip.
2. Run the Python script: 'Fashion_MNIST_python.ipynb'
3. The script will train the CNN model and display predictions for two test images.

### R
1. Download the zipped file ('Fashion_MNIST_CNN.zip') and unzip.
2. Open 'Fashion_MNIST_Assignment_R_Script.R' in RStudio.
3. Run the script line-by-line or execute it fully.
-  The R script will train the CNN model and display predictions for two test images.

## Steps in the Code
1. Import Libraries: The necessary libraries are imported to handle data, visualize results, and build the CNN model
2. Load the Dataset: The Fashion MNIST dataset is loaded using TensorFlow's Keras API
3. Preprocess the Data:
   
-  Normalization: The pixel values of the images are scaled to a range between 0 and 1 by dividing by 255.
-  Reshaping: The images are reshaped to include a single color channel to match the CNN input requirements.
-  Categorical Conversion: The labels are converted to one-hot encoded format using to_categorical.
  
4. Build the CNN Model: The model is built using the Keras Sequential API. It includes convolutional layers for feature extraction, max-pooling layers for downsampling, and dense layers for classification
5. Compile the Model: The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric
6. Train the Model: The model is trained on the training dataset for 10 epochs, with validation on the test dataset
7. Evaluate the Model: After training, the model is evaluated on the test dataset to check its accuracy
8. Make Predictions: The model predicts the categories of the first two images from the test dataset and compares them with the actual labels. The predictions are visualized
9. Save model for documentation

## Outputs
After running the scripts, the following outputs will be generated:
- Training Metrics: Accuracy and loss for both training and validation datasets.
- Test Evaluation: Model performance (accuracy) on the test dataset.
- Predictions: Visualizations of two test images with predicted and actual labels.

## Notes
- Ensure that TensorFlow and Keras are properly installed for both Python and R.
- Adjust the number of epochs and batch size in the code for experimentation.
  
