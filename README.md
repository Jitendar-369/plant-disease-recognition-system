# Plant Disease Recognition Using CNN

## Overview

This project uses a Convolutional Neural Network (CNN) to classify plant leaf images into different categories, such as Healthy, Powdery Mildew, and Rust. The model is trained on a dataset of labeled images, and the data augmentation is applied to improve model generalization. This repository contains the code to load, preprocess, train, evaluate, and use the model for plant disease classification.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Results](#results)
6. [Usage](#usage)
7. [License](#license)

## Installation

To use this project, follow the steps below:

### Clone the repository:
```bash
git clone https://github.com/your_username/plant-disease-recognition-cnn.git
cd plant-disease-recognition-cnn
Install dependencies:

pip install -r requirements.txt

requirements.txt contains:

shell
Copy
Edit
tensorflow>=2.0
numpy
matplotlib
pandas
seaborn
scikit-learn

Dataset
The dataset should be organized in the following structure:


/Plant Disease Recognition
    /Train
        /Healthy
        /Powdery
        /Rust
    /Validation
        /Healthy
        /Powdery
        /Rust
    /Test
        /Healthy
        /Powdery
        /Rust
Make sure to update the data_dir path in the code to point to the location of your dataset in Google Drive or local machine.

Model Architecture
The CNN model is built with the following layers:

Conv2D Layer 1: 32 filters, kernel size (3, 3), ReLU activation
MaxPooling2D Layer 1: Pooling size (2, 2)
Conv2D Layer 2: 64 filters, kernel size (3, 3), ReLU activation
MaxPooling2D Layer 2: Pooling size (2, 2)
Dropout Layer 1: Dropout rate 0.25 for regularization
Conv2D Layer 3: 128 filters, kernel size (3, 3), ReLU activation
MaxPooling2D Layer 3: Pooling size (2, 2)
Dropout Layer 2: Dropout rate 0.25 for regularization
Flatten Layer: Flatten the feature map for the dense layers
Dense Layer 1: 256 units, ReLU activation, Dropout rate 0.5
Dense Layer 2: 3 units for classification (Healthy, Powdery Mildew, Rust), Softmax activation
Model Compilation
The model is compiled using:

Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Training
Run the following code to start training the model:

python
Copy
Edit
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)
The training process includes early stopping to prevent overfitting.

Results
Once trained, you can visualize the model's performance with accuracy and loss plots. The training and validation accuracy are plotted over epochs to monitor progress.

python
Copy
Edit
# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
Usage
Once the model is trained, you can use it to predict plant disease on new images. Here's how you can make predictions:

Preprocess the image:
python
Copy
Edit
img_array = preprocess_input_image(img_path)
Make predictions:
python
Copy
Edit
predicted_class = predict_disease(model, img_array)
print(f'The predicted class is: {predicted_class}')
Make sure to replace img_path with the path to the image you want to classify.

License
This project is licensed under the MIT License - see the LICENSE file for details.

yaml
Copy
Edit

---

Feel free to customize this further according to your specific needs! Just copy-paste it into your `README.md` file in the repository. Let me know if you need anything else!
