#if you are using sspcloud you must pip install matplotlib in an interactive window
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

# CHECK GPU or choose CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



####### DATA PREPARATION #####

transform = transforms.Compose([
    transforms.ToTensor()])

try:
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")


# CHOOSE THE AMOUNT OF NOISE TO ADD TO THE IMAGES
noise_mean = 0.0
noise_std = 0.1

def add_gaussian_noise(tensor, noise_mean, noise_std):
    #noise = torch.randn(tensor.size(),device=tensor.device) * noise_std + noise_mean
    noise = torch.randn(tensor.size()) * noise_std + noise_mean
    return tensor + noise


######## DEFINE THE MODEL ########

###### Define a class for the small_mnist_model #####   # we use classes to export and import the model easily

# Smaller model for MMNIST dataset, this is the same model as big_model but with less intermediate convolution layers

class Small_MNIST(nn.Module):

    def __init__(self):
        super(Small_MNIST, self).__init__()
        self.model = nn.Sequential(
            # ADD A REVERSIBLE DOWN SAMPLING
            # Here we use the reverse of pixel shuffle to downsample an image into 4 images
            nn.PixelUnshuffle(2),
            # First convolution layer
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 2 convolution layers with batch normalization and ReLU
            *[nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)) for _ in range(2)],
            # Final convolution layer
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, padding=1),
            # ADD THE REVERSE OF THE DOWN SAMPLING
            nn.PixelShuffle(2)
)
    def forward(self, x):
        return self.model(x)


#small_mnist_model = Small_MNIST().to(device)
small_mnist_model = Small_MNIST()
# Print the model architecture
print(f"small_mnist_model architecture: {small_mnist_model}")



######## TRAINING ########
########### TRAINING OF THE small_mnist_model ######

#### The code doesn't run on my laptop, I don't have enough memory to run it. Use sspcloud with GPU to run it and make sure to allocate enough memory

# approximately 3 min to train on sspcloud
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(small_mnist_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    small_mnist_model.train()
    running_loss = 0.0
    for images, _ in trainloader:
        # Move images to the appropriate device
        #images = images.to(device)

        # Add Gaussian noise to the images
        noisy_images = add_gaussian_noise(images, noise_mean, noise_std)

        # Reset the gradient
        optimizer.zero_grad()

        # Forward
        outputs = small_mnist_model(noisy_images)

        # Compute the loss
        loss = criterion(outputs, images)

        # Backward propagation
        loss.backward()

        # Optimization step, update the parameters of the model
        optimizer.step()

        # Add value of the loss for the current batch for the epoch
        running_loss += loss.item()

    # Compute the average loss of the epoch
    average_loss_epoch = running_loss / len(trainloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss_epoch:.4f}')

    # Step the scheduler
    scheduler.step(average_loss_epoch)

print("Training complete")




#### WE SAVE THE MODEL SO WE DON'T HAVE TO TRAIN IT AGAIN ####

# Create the directory where all models will be stored if the directory doesn't already exist
os.makedirs('trained_models', exist_ok=True)
# Define the file paths
model_path = f'trained_models/small_mnist_model_{num_epochs}_epochs_{noise_std}_noise_std.pth'
optimizer_path = f'trained_models/optimizer_mnist_model_{num_epochs}_epochs_{noise_std}_noise_std.pth'
# Save the model state dictionary
torch.save(small_mnist_model.state_dict(), model_path)

# Save the optimizer state dictionary
torch.save(optimizer.state_dict(), optimizer_path)

print("Model and optimizer states have been saved.")
