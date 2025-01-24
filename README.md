CIFAR-10 Image Classification Model
I developed this project to perform image classification using CIFAR-10 dataset. I used transfer learning with pre-trained VGG16 model and enriched it with custom layers. Below is a detailed description of my project.

Project Purpose
The purpose of this project is to create a deep learning based classification model to classify images in CIFAR-10 dataset into 10 distinct classes:
1. Airplane
2. Car
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

CIFAR-10 Dataset
• Size: 60,000 32x32 color images (50,000 training, 10,000 testing).
• Classes: 10 different classes with 6,000 images per class.
• Usage: Image classification and computer vision

Project Steps

1. Data Preprocessing
• The dataset was loaded using TensorFlow Keras and split into training and test sets.
• The pixel values ​​of the images were normalized to the range [0, 1], which increased the learning speed and stability of the model.
• The labels were converted to one-hot encoded format for classification purposes.

2. Model Architecture
The project uses transfer learning:
• VGG16: A model pre-trained on the "ImageNet" dataset. The upper layers were removed and the weights were frozen (untrainable).
• Custom Layers:
◦ Flatten: Flattened the output from VGG16.
◦ Dense Layers: Two fully connected layers with 512 and 256 neurons using ReLU activation.
◦ Dropout: 30% and 50% dropout rates were used to reduce overfitting.
◦ Output Layer: A softmax activation to classify into 10 categories.

3. Model Compilation
• Optimizer: Adam optimizer with low learning rate (0.0001) was used.
• Loss Function: “Categorical cross entropy” was used for multi-class classification.
• Metric: Accuracy was monitored to evaluate the performance.

4. Model Training
• The model was trained for 10 epochs with a batch size of 32.
• Validation performance was evaluated at the end of each epoch using the test set.

5. Model Evaluation
• Predictions: Class predictions were made on the test set.
• Classification Report: Precision, recall and F1 score metrics were calculated for each class.
• Confusion Matrix: Correct and incorrect classifications were visualized.

6. Visualization
• Training and validation accuracy and loss curves were plotted to analyze the learning progress of the model.

Results
• Training Accuracy: 97.68%
• Validation Accuracy: 85.60%
• Test Accuracy: 85.60%


Technologies
• TensorFlow/Keras
• Python
• Matplotlib
• Scikit-learn
