Sure! Here's the updated `README.md` that includes instructions on downloading the MVTec AD dataset and referencing the research paper:

---

# Autoencoder-based Image Reconstruction for Fault Detection

This repository demonstrates an implementation of an autoencoder-based model for reconstructing images and detecting faults in carpets. The dataset used contains images of carpets that are either "good" or have some form of fault. The goal of this project is to train an autoencoder model to reconstruct images and compute reconstruction errors to identify faults in unseen carpet images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Fault Detection](#fault-detection)
- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Research Paper](#research-paper)

## Installation

To run the code, first, clone the repository and install the required dependencies. You can use the following commands:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

### Requirements

- `torch` (PyTorch)
- `torchvision`
- `matplotlib`
- `PIL`
- `tqdm`
- `seaborn`
- `scikit-learn`

## Usage

### 1. Loading the Dataset

The dataset is loaded using `ImageFolder` from `torchvision.datasets`, and images are transformed to tensors with resizing for consistent input size to the model.

```python
from torchvision.datasets import ImageFolder
train_image_path = "carpet/train"
good_dataset = ImageFolder(root=train_image_path, transform=transform)
```

The dataset is then split into training and testing sets using `torch.utils.data.random_split`.

### 2. Defining the Autoencoder Model

The core of the project is an autoencoder architecture. It consists of two main parts:

- **Encoder**: Reduces the dimensionality of the input image by applying convolutional layers and pooling operations.
- **Decoder**: Reconstructs the image from the encoded feature representation using transposed convolutional layers.

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

### 3. Training the Model

The model is trained using Mean Squared Error (MSE) as the loss function and Adam optimizer.

```python
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
```

The training loop processes the images in batches, computes the loss, and updates the weights accordingly.

```python
for epoch in tqdm(range(num_epochs)):
    model.train()
    for img, _ in train_loader:
        output = model(img)
        loss = criterion(output, img)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
```

The training and validation loss are plotted after the training is completed.

### 4. Fault Detection

Fault detection is performed by computing the reconstruction error for each image. If the reconstruction error exceeds a predefined threshold, the image is classified as containing a fault.

The threshold is determined based on the distribution of reconstruction errors in the training set. The model calculates the squared mean of the reconstruction error across the image and uses it for classification.

```python
data_recon_squared_mean = ((data - recon) ** 2).mean(axis=1)[:, 0:-10, 0:-10].mean(axis=(1, 2))
```

### 5. Evaluation

The model's performance is evaluated using ROC curve and AUC score to assess the quality of fault detection.

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc_roc_score = roc_auc_score(y_true, y_score)
```

## Model Architecture

- **Encoder**: 
  - Consists of 3 convolutional layers followed by ReLU activations and average pooling layers.
  
- **Decoder**: 
  - Uses transposed convolutional layers to upsample and reconstruct the image.

- **Loss Function**:
  - The model is trained using Mean Squared Error (MSE) between the input and reconstructed images.

## Training and Evaluation

- **Training Loop**: 
  - The model is trained for `100` epochs. Loss is calculated for each batch and epoch.
  
- **Validation**:
  - After every epoch, validation loss is computed on the test set.
  
- **Fault Detection**:
  - The fault detection is based on the reconstruction error, and a threshold is set to classify images as faulty or not.

## Dataset

The dataset used in this project is the **MVTec Anomaly Detection (MVTec AD) dataset**. It contains images of industrial objects, including carpets, with normal and faulty samples. To download the dataset, visit the following link:

- **MVTec AD Dataset**: [Download MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

This dataset contains images of objects in different categories with defects such as scratches, dents, and more, which are useful for evaluating fault detection systems like the autoencoder model in this project.

## Research Paper

For further reference and to understand the methods and techniques used in this approach, you can refer to the following research paper:

- **Paper**: [Anomaly Detection with Autoencoders for Industrial Applications](https://arxiv.org/pdf/2012.07122)

This paper provides insights into the application of autoencoders for anomaly detection in industrial settings and lays the foundation for techniques used in this project.

## Conclusion

This project demonstrates how an autoencoder can be trained to reconstruct images and detect faults based on the reconstruction error. The model can be further fine-tuned and extended for more complex applications in fault detection tasks.

---