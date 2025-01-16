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

from every_model_architecture import FFDNet_inspired_small_mnist  # Import the FFDNet_inspired_small_mnist model class defined in an other file

# CHECK GPU or choose CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#### IMPORTANT TO NOTE THAT FOR THE MOMENT WE ALWAYS USE THE SAME NOISE MAP FOR ALL IMAGES ####
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


######### IMPORT THE MODEL ########
model = FFDNet_inspired_small_mnist
# Print the model architecture
print(f"FFDNet_inspired_small_mnist architecture: {FFDNet_inspired_small_mnist}")



######## TRAINING ########

#### The code doesn't run on my laptop, I don't have enough memory to run it. Use sspcloud with GPU to run it and make sure to allocate enough memory

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, _ in trainloader:
        # Move images to the appropriate device
        #images = images.to(device)

        # Add Gaussian noise to the images
        noisy_images = add_gaussian_noise(images, noise_mean, noise_std)

        # Reset the gradient
        optimizer.zero_grad()

        # Forward
        outputs = model(noisy_images)

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
model_path = f'trained_models/FFDNet_inspired_small_mnist_{num_epochs}_epochs_{noise_std}_noise_std.pth'
optimizer_path = f'trained_models/optimizer_FFDNet_inspired_small_mnist{num_epochs}_epochs_{noise_std}_noise_std.pth'
# Save the model state dictionary
torch.save(model.state_dict(), model_path)

# Save the optimizer state dictionary
torch.save(optimizer.state_dict(), optimizer_path)

print("Model and optimizer states have been saved.")