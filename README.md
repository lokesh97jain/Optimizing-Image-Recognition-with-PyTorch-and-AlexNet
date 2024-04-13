# Optimizing-Image-Recognition-with-PyTorch-and-AlexNet

This project explores the construction and optimization of neural networks for image classification tasks, focusing on the AlexNet architecture. It includes building a basic neural network, optimizing it, implementing and improving AlexNet, and enhancing model performance with data augmentation.

<img src="https://github.com/lokesh97jain/Optimizing-Image-Recognition-with-PyTorch-and-AlexNet/assets/44944748/3c05285a-0d53-4854-b28e-ba5890e3569f" height="50%" width="50%">

## Overview
The project covers several key areas:
  1. **Building a Basic Neural Network**: Begins with a foundational approach to binary classification, highlighting data preprocessing, model architecture, and performance metrics.
  2. **Optimizing the Neural Network**: Investigates the impact of hyperparameter tuning, including dropout rates, activation functions, and optimizers. Techniques such as Early Stopping and K-Fold Cross Validation are employed to enhance model performance.
  3. **Implementing & Improving AlexNet**: Adapts AlexNet to classify images into categories such as dogs, vehicles, and food, achieving significant accuracy.
  4. **Optimizing CNN & Data Augmentation**: Utilizes the Street View House Numbers (SVHN) dataset, applying data augmentation to improve the model's generalization and performance.

## Folder Structure

Dataset Link:
```
https://kaggle.com/datasets/c3a1ad7d0ab948e6bf9f2242ae06247ed5ff8adc894215aa1a292992ea9d99bc
```

```
├── datasets/                      # Folder containing dataset used in the project
│   ├── dogs/          
│   ├── foods/     
│   └── vehicles/
```

Python Code :
```
AlexNet_Image_Classification_Project/           
│
├── basic_NN_construction.ipynb/                     # Jupyter notebooks with all coding experiments
├── advanced_NN_optimization_experiments.ipynb/ 
├── alexnet_implementation.ipynb/ 
└── alexnet_optimization_and_data_augmentation.ipynb/ 
```

## Project Structure

  * **basic_NN_construction.ipynb**: Details the construction and initial experimentation of the basic neural network.
  * **advanced_NN_optimization_experiments.ipynb**: Explores further experiments and optimizations.
  * **alexnet_implementation.ipynb**: Demonstrates AlexNet implementations and optimizations.
  * **alexnet_optimization_and_data_augmentation.ipynb**: Advanced optimization and data augmentation (VGG13).

## Getting Started
To replicate our findings or build upon them, ensure you have the following:

  * Python 3.x
  * TensorFlow
  * Keras
  * NumPy
  * Matplotlib for plotting

``` bash
pip install tensorflow keras numpy matplotlib
```

## Running the Project

  1. Clone this repository.
  2. Ensure the necessary dependencies are installed.
  3. Follow the notebooks sequentially to grasp the flow from data preprocessing to model optimization.

## Data Preparation
The project utilizes diverse datasets, including images of dogs, vehicles, food, and street view house numbers. Key steps in data preparation include:

  * Normalization: Pixel values are normalized to have a mean of 0 and standard deviation of 1, facilitating model training and convergence.
  * Augmentation: To enhance model robustness, data augmentation techniques like random rotations, resizing, and normalization are applied, creating a more diverse training set.

## Model Architecture

#### Basic Neural Network
The initial model is a straightforward neural network designed for binary classification, featuring layers with ReLU and sigmoid activation functions.

#### AlexNet Adaptation
The architecture mimics the original AlexNet with adjustments for the specific datasets used. It includes convolutional layers, max-pooling, and fully connected layers, employing ReLU activations and dropout for regularization.

#### Optimizations
  * Early Stopping: Monitors validation loss to halt training preemptively, preventing overfitting.
  * K-Fold Cross Validation: Ensures model reliability and generalizability across different data subsets.
  * Learning Rate Scheduler: Adjusts the learning rate dynamically, optimizing the training phase.

#### VGG13 
VGG-13 is a convolutional neural network model known for its simplicity, using 13 layers with weights (including convolutional and fully connected layers) that was developed for large-scale image recognition tasks.

## Results
Adapted AlexNet reached a peak test accuracy of 90.18% and a training loss of 0.3100, demonstrating outstanding generalization to unseen data.

