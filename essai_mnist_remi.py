import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch
from torchvision import datasets, transforms


################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################
################ FICHIER POUBELLE ################


# CHECK GPU or choose CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




##### DATA PREPARATION #####
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



##### PLOT THE FIRST 5 clean images
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i in range(5):
    ax = axes[i]
    ax.imshow(images[i].numpy().squeeze(), cmap="gray")
    ax.set_title(f'{labels[i].item()}')
    ax.axis("off")
fig.suptitle("First five original clean images from MNIST", fontsize=16)
plt.show()


# Save the original clean dataset
train_clean_images = images
train_label = labels

# Create noisy images by adding AWGN

# CHOOSE THE AMOUNT OF NOISE
noise_mean = 0.0
noise_std = 0.1


def add_gaussian_noise(tensor, noise_mean=0.0, noise_std=1.0):
    #noise = torch.randn(tensor.size(),device=tensor.device) * noise_std + noise_mean
    noise = torch.randn(tensor.size()) * noise_std + noise_mean
    return tensor + noise





transform_add_noise = transforms.Compose([
    transforms.Lambda(lambda x: add_gaussian_noise(x, noise_mean, noise_std))
])

train_noisy_images = transform_add_noise(train_clean_images)

##### PLOT THE FIRST 5 noisy images
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i in range(5):
    ax = axes[i]
    ax.imshow(train_noisy_images[i].numpy().squeeze(), cmap="gray")
    ax.set_title(f'{labels[i].item()}')
    ax.axis("off")
fig.suptitle("First five noised images from MNIST", fontsize=16)
plt.show()



#### Plot the first 5 clean and noised images
fig, axes = plt.subplots(2, 5, figsize=(15, 4))
for i in range(5):
    # Plot clean images on the first
    ax = axes[0, i]
    ax.imshow(images[i].numpy().squeeze(), cmap="gray")
    ax.set_title(f'{labels[i].item()}')
    ax.axis("off")

    # Plot the noised version of the images on the second row
    ax = axes[1, i]
    ax.imshow(train_noisy_images[i].numpy().squeeze(), cmap="gray")
    ax.set_title(f'{labels[i].item()}')
    ax.axis("off")

# Add a title to the entire figure
fig.suptitle("Original clean and noised version of MNIST images", fontsize=16)
plt.show()



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




##### LOAD THE MODEL #####
# Reinitialize the model and optimizer
small_mnist_model = Small_MNIST()  # Replace with your model class
optimizer = optim.Adam(small_mnist_model.parameters(), lr=0.001)  # Use the same optimizer and parameters

# Load the model state dictionary
small_mnist_model.load_state_dict(torch.load('small_mnist_model.pth'))

# Load the optimizer state dictionary
optimizer.load_state_dict(torch.load('optimizer.pth'))

# Set the model to evaluation mode
small_mnist_model.eval()

print("The small_mnist_model and optimizer states have been loaded.")



###### TEST ######

# Test the model on the first 5 images of the dataset



# Load the MNIST dataset with the custom transform
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

# Get the first 5 images from the test dataset
dataiter = iter(testloader)
images, labels = next(dataiter)

# Add Gaussian noise to the images
noisy_images = add_gaussian_noise(images, noise_mean=0,noise_std=1).to(device)




# Run the model on the noisy images
with torch.no_grad():
    denoised_images = small_mnist_model(noisy_images)

# Plot the original, noisy, and denoised images
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i in range(5):
    # Plot original images in the first row
    ax = axes[0, i]
    ax.imshow(images[i].cpu().numpy().squeeze(), cmap="gray")
    ax.set_title(f'Original {labels[i].item()}')
    ax.axis("off")

    # Plot noisy images in the second row
    ax = axes[1, i]
    ax.imshow(noisy_images[i].cpu().numpy().squeeze(), cmap="gray")
    ax.set_title(f'Noisy {labels[i].item()}')
    ax.axis("off")

    # Plot denoised images in the third row
    ax = axes[2, i]
    ax.imshow(denoised_images[i].cpu().numpy().squeeze(), cmap="gray")
    ax.set_title(f'Denoised {labels[i].item()}')
    ax.axis("off")

# Add a title to the entire figure
fig.suptitle("Original, Noisy, and Denoised MNIST Images", fontsize=16)
plt.show()