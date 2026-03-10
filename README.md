# EX-4 : DL- DEVELOPING A NEURAL NETWORK CLASSIFICATION MODEL USING TRANSFER LEARNING

#### Name:R.JAYASREE
#### Register Number:212223040074
## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
The problem statement for this experiment is to develop an image classification model that can accurately distinguish between 'defect' and 'notdefect' semiconductor chip images. This is a binary classification task, where the goal is to leverage transfer learning using a pre-trained VGG19 model to effectively classify new, unseen chip images.

## Neural Network Model

<img width="1248" height="963" alt="image" src="https://github.com/user-attachments/assets/3a96e18b-b882-4222-974f-f2882a54b818" />


## DESIGN STEPS


### STEP 1:
Data Loading and Preprocessing: Load the chip image dataset, apply necessary transformations like resizing and converting to tensors, and create data loaders for efficient batch processing. This step also includes visualizing sample images and checking dataset statistics.

### STEP 2: 
Model Setup for Transfer Learning: Load a pre-trained VGG19 model, modify its final classification layer to match the binary classification task (defect/not defect), freeze the convolutional layers to retain pre-trained features, and define the loss function and optimizer.

### STEP 3: 
Model Training: Train the modified VGG19 model using the prepared training data. The training process will involve iterating through epochs, calculating loss, performing backpropagation, and updating the model's weights. Training and validation loss will be tracked and plotted.

### STEP 4: 
Model Evaluation and Reporting: Evaluate the trained model's performance on the test dataset. This includes calculating and displaying the test accuracy, generating and visualizing a confusion matrix, and printing a detailed classification report.

### STEP 5:
Single Image Prediction: Demonstrate the model's predictive capability by performing inference on individual images from the test dataset and displaying the actual and predicted labels along with the image.


## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import  VGG19_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])

!unzip -qq ./chip_data.zip -d data

dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

show_sample_images(train_dataset)
print(f"Total number of training samples: {len(train_dataset)}")

first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

print(f"Total number of training samples: {len(test_dataset)}")

first_image, label = test_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model=models.vgg19(weights=VGG19_Weights.DEFAULT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
summary(model, input_size=(3, 224, 224))
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for param in model.features.parameters():
    param.requires_grad = False 
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train_model(model, train_loader,test_loader,num_epochs=10):
    # Write your code here
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
      running_loss = 0.0
      for images , labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      train_losses.append(running_loss / len(train_loader))
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_labels.unsqueeze(1).float())
            val_loss += loss.item()
      val_losses.append(val_loss / len(test_loader))
      model.train()

      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    print("Name: Jayasree R")
    print("Register Number:212223040074")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_model(model, train_loader,test_loader,num_epochs=10)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    print("Jayasree R")
    print("Register Number: 212223040074")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print("Name:Jayasree R")
    print("Register Number:212223040074")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

test_model(model, test_loader)

def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().item()


    class_names = class_names = dataset.classes
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

predict_image(model, image_index=55, dataset=test_dataset)

predict_image(model, image_index=25, dataset=test_dataset)


```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="863" height="719" alt="image" src="https://github.com/user-attachments/assets/4f19bf1c-70c1-41e8-9acb-198b16a27cf5" />


### Confusion Matrix

<img width="901" height="745" alt="image" src="https://github.com/user-attachments/assets/ce74dcfb-b936-4313-97f8-8b30194ecf6b" />



### Classification Report

<img width="626" height="244" alt="image" src="https://github.com/user-attachments/assets/08a84853-2a0f-4203-afad-1e742c8af4ec" />


### New Sample Data Prediction

<img width="569" height="581" alt="image" src="https://github.com/user-attachments/assets/b6c2daa0-5fb8-44d5-83e4-af639168342b" />


<img width="552" height="567" alt="image" src="https://github.com/user-attachments/assets/a504d7e2-aabc-4a56-98e3-00b30078a4bb" />


## RESULT
Thus the Program for develop an image classification model that can accurately distinguish between 'defect' and 'notdefect' semiconductor chip images is executed successfully.
