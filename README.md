Plant Disease Recognition Using CNN
Overview
This project implements a Convolutional Neural Network (CNN) model for recognizing plant diseases from images. The model classifies plant leaf images into different categories (such as Healthy, Powdery Mildew, and Rust). The dataset is divided into training, validation, and test sets, with data augmentation applied to the training set for better generalization.

Table of Contents
Installation
Dataset
Model Architecture
Training
Results
Usage
License
Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your_username/plant-disease-recognition-cnn.git
cd plant-disease-recognition-cnn
Install required dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Here is the content for requirements.txt:

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
The dataset contains images of plant leaves labeled with disease types. It is divided into the following directories:

Train: Contains the training images grouped by their class (Healthy, Powdery Mildew, Rust).
Validation: Contains the validation images for tuning the model.
Test: Contains test images for final evaluation.
Make sure to update the data_dir path in the code to point to the location of your dataset on Google Drive.

Model Architecture
The CNN model consists of the following layers:

Conv2D Layer 1: 32 filters, kernel size of (3, 3), ReLU activation
MaxPooling2D Layer 1: Pooling size of (2, 2)
Conv2D Layer 2: 64 filters, kernel size of (3, 3), ReLU activation
MaxPooling2D Layer 2: Pooling size of (2, 2)
Dropout Layer 1: Dropout rate of 0.25 for regularization
Conv2D Layer 3: 128 filters, kernel size of (3, 3), ReLU activation
MaxPooling2D Layer 3: Pooling size of (2, 2)
Dropout Layer 2: Dropout rate of 0.25 for regularization
Flatten Layer: Flatten the feature map for the dense layers
Dense Layer 1: 256 units, ReLU activation, Dropout rate of 0.5
Dense Layer 2: 3 units (for the three classes), Softmax activation
Model Compilation
The model is compiled using:

Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Training
To train the model, run the following code:

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
Once trained, the model achieves high accuracy in classifying plant diseases. The results can be evaluated on the validation and test datasets. The confusion matrix and classification report will help in understanding the model's performance.

Accuracy Plot
Training and validation accuracy are plotted over the epochs for performance visualization.

python
Copy
Edit
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
Loss Plot
Training and validation loss are plotted over the epochs to monitor the model's training progress.

python
Copy
Edit
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
Usage
Once the model is trained, you can use it to predict plant disease on new images.

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
Make sure to replace img_path with the path to the test image.

License
This project is licensed under the MIT License - see the LICENSE file for details.
