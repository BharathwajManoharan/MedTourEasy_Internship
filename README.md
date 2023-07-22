# American Sign Language (ASL) Letter Classification

American Sign Language (ASL) is the primary language used by many deaf individuals in North America, and it is also used by hard-of-hearing and hearing individuals. This project aims to build a convolutional neural network (CNN) to classify images of ASL letters. The CNN will be trained to recognize individual letters, which can be a first step towards developing a sign language translation system.

## Dataset

The dataset consists of images of ASL letters. Each image is a hand sign representing a specific letter, such as 'A', 'B', 'C', and so on. The images are preprocessed and split into training and test sets.

## Visualizing the Data

We begin by visualizing the training data to get an overview of the ASL letter images in the dataset. The images will be displayed along with their corresponding labels (letters).

## Examining the Dataset

We examine the distribution of images for each letter in both the training and test sets. This allows us to verify that the dataset has roughly equal proportions of each letter, which is essential for training a balanced model.

## One-Hot Encoding

Before training the CNN, we one-hot encode the categorical labels, converting them into a format suitable for use in the Keras model.

## Model Architecture

The CNN architecture is defined in Keras. It consists of convolutional layers, max-pooling layers, and dense layers. The output layer uses the softmax activation function to predict the probabilities for each letter.

## Model Compilation

We compile the CNN model by specifying the optimizer and loss function. We use the Adam optimizer and categorical cross-entropy loss for multi-class classification.

## Model Training

The model is trained using the training dataset. The training process involves iterating over batches of data for a certain number of epochs. We also use a validation split to monitor the model's performance during training.

## Model Evaluation

Once the model is trained, we evaluate its performance on the test dataset to see how well it generalizes to unseen data.

## Visualizing Mistakes

We examine the images that were misclassified by the model. This allows us to gain insights into the model's weaknesses and potential areas for improvement.

Feel free to explore the code and results in this repository! If you have any questions or suggestions, please let us know.

## How to Use

To run the code, you will need Python, TensorFlow, and Keras installed. You can install the required packages using the following command:

```bash
pip install tensorflow keras matplotlib
```

## Credits
This project was developed as part of an internship under Medtoureasy . The dataset used in this project is obtained from [Here](https://drive.google.com/uc?export=download&id=1o3Eu6DLIc2UYSV0dkUML8BUxsHfBE68J).
