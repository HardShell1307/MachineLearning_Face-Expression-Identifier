# Face Expression Identifier using ResNet18

This project implements a face expression identifier using the ResNet18 architecture, which is trained on the FERCDataset available on Kaggle. The goal is to build a model that can classify facial expressions into one of seven categories: anger, disgust, fear, happiness, neutral, sadness, and surprise.

## Dataset

The dataset used for this project is the FERCDataset, which consists of labeled facial expression images. The dataset is divided into two main folders: "train" and "test," containing training and testing images, respectively. Each of these folders contains subfolders for each class (emotion), where the actual images are stored.

Dataset Link: https://www.kaggle.com/manishshah120/facial-expression-recog-image-ver-of-fercdataset

## Model Architecture

The model architecture used for this project is ResNet18, which is a popular deep learning architecture. The ResNet18 architecture is pre-trained on the ImageNet dataset and then fine-tuned for face expression recognition by replacing the fully connected layer with a new layer having seven output nodes (one for each class).

## Data Augmentation

Data augmentation is applied to the training dataset using various transformations such as random horizontal flipping, random rotation, and color jitter. These augmentations help the model generalize better and improve its performance on unseen data.

## Training

The model is trained using the "fit_one_cycle" training method, which implements the one-cycle learning rate policy. This policy adjusts the learning rate during training, starting with a lower learning rate, gradually increasing it, and then decreasing it. This technique often leads to faster convergence and better performance.

## Evaluation

The model's performance is evaluated using the validation dataset after each epoch. The evaluation metrics used are cross-entropy loss and accuracy. The validation loss and accuracy are recorded for each epoch to monitor the model's performance during training.

## Plots

Several plots are generated to visualize the training process and model performance. These plots include:

1. **Accuracy vs. Number of Epochs:** This plot shows how the accuracy of the model changes as the number of epochs increases during training.

2. **Training and Validation Losses:** This plot displays the training and validation losses for each epoch. It helps monitor the convergence and overfitting of the model.

3. **Learning Rate vs. Number of Epochs:** This plot illustrates how the learning rate changes over epochs during training. The one-cycle learning rate policy can be observed from this plot.

## Usage

To run the code and train the face expression identifier, follow these steps:

- Clone the repository:
  ```bash
  git clone https://github.com/your_username/face-exp-resnet.git
  ```
- Download the FERCDataset from Kaggle and place it in the appropriate directory.
  
- Install the required dependencies:
  ```bash
  pip install torch torchvision matplotlib tqdm jovian
  ```
- Run the training script :
  ```bash
  python train.py
  ```
- Check the results and performance in the output and generated plots.

## Acknowledgments
- The FERCDataset used in this project is sourced from Kaggle.
- The ResNet18 model architecture is based on torchvision's implementation.
