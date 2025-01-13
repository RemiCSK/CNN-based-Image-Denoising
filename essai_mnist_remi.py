import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch
from torchvision import datasets, transforms



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




# The big_model is probably too complex for the MNIST dataset.
# The receptive field is probably way too big (bigger than the images)
# So we should create a small model (less layers) for the MNIST dataset


# CNN ARCHITECTURE OF THE BIG MODEL WITH 15 CONVOLUTION LAYERS
big_model = nn.Sequential(

    # ADD A REVERSIBLE DOWN SAMPLING


    # First convolution layer
    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),

    # 13 convolution layers with batch normalization and ReLU
    *[nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    ) for _ in range(13)],

    # Final convolution layer
    nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, padding=1)

    # ADD A REVERSIBLE UP SAMPLING

)

# Print the model architecture
print(big_model)



########### TRAINING ###### Don't run for the moment because no down sampling and up sampling and model too complex

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(big_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    big_model.train()
    running_loss = 0.0
    for images, _ in trainloader:
        # Add Gaussian noise to the images
        noisy_images = add_gaussian_noise(images, noise_mean, noise_std)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = big_model(noisy_images)

        # Compute the loss
        loss = criterion(outputs, images)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item()

    # Compute the average loss for this epoch
    avg_loss = running_loss / len(trainloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Step the scheduler
    scheduler.step(avg_loss)

print("Training complete")