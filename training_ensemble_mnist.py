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

# Import models
from every_model_architecture import FFDNet_inspired_small_mnist
from every_model_architecture import FFDNet_inspired_small_mnist_extend
from every_model_architecture import FFDNet_mnist_ensemble


# CHECK GPU or choose CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#### IMPORTANT TO NOTE THAT FOR THE MOMENT WE ALWAYS USE THE SAME NOISE MAP FOR ALL IMAGES ####
####### DATA PREPARATION #####

transform = transforms.Compose([
    transforms.ToTensor()])

try:
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset)//6, shuffle=False)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")



def add_gaussian_noise(tensor, noise_mean, noise_std):
    noise = torch.randn(tensor.size()) * noise_std + noise_mean
    return tensor + noise



# CREATE A UNIFORM NOISE LEVEL MAP THAT MATCHES DIMENSIONS OF DOWNSAMPLED IMAGES
def create_uniform_noise_level_map(batch_size, height, width, noise_std):
    noise_level_map = torch.full((batch_size, 1, height // 2, width // 2), noise_std) # Creates noise level map with only noise_std values in it.
    return noise_level_map

# create different uniform noise level map with noise_std from 0.05 to 0.3 with step 0.05 thus creating 6 different noise level maps for each image for the training. We have 6 times more training images.


# IMPORT MODELS
model1 = FFDNet_inspired_small_mnist()
model2 = FFDNet_inspired_small_mnist_extend()
model = FFDNet_mnist_ensemble(model1, model2)
print(f"Architecture model 1 : {model1}")
print(f"Architecture model 2 : {model2}")

print(f"FFDNet_mnist_ensemble architecture: {model}")




######## TRAINING ########

#### The code doesn't run on my laptop, I don't have enough memory to run it. Use sspcloud with GPU to run it and make sure to allocate enough memory

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)

noise_mean = 0.0
noise_std_vector = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]   # We will train the model with 6 different uniform noise level maps
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, _ in trainloader:
        noisy_images = []
        noise_level_maps = []
        for noise_std in noise_std_vector:
         # Add Gaussian noise to the images
            noisy_im = add_gaussian_noise(images, noise_mean, noise_std)

            # Generate CORRESPONDING noise level map
            b, _, h, w = images.size() # makes sure good dimension
            noise_l_map = create_uniform_noise_level_map(b, h, w, noise_std)

            # Append to the images with other noise levels
            noisy_images.append(noisy_im)
            noise_level_maps.append(noise_l_map)


        # Concatenate ALL noisy images and noise level maps
        noisy_images = torch.cat(noisy_images, dim=0)
        noise_level_maps = torch.cat(noise_level_maps, dim=0)

        # Repeat the original images to match size of the concatenated noisy images (we have 6 times more noisy images so we need 6 times more true images)
        images_repeated = images.repeat(len(noise_std_vector), 1, 1, 1)


        # Reset the gradient
        optimizer.zero_grad()

        # Forward
        outputs = model(noisy_images, noise_level_maps)

        # Compute the loss
        loss = criterion(outputs, images_repeated)

        # Backward propagation
        loss.backward()

        # Optimization step, update the parameters of the model
        optimizer.step()

        # Add value of the loss for the current batch for the epoch
        running_loss += loss.item()

    # Compute the average loss of the epoch
    average_loss_epoch = running_loss / (len(trainloader) * len(noise_std_vector))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss_epoch:.4f}')

    # Step the scheduler
    scheduler.step(average_loss_epoch)

print("Training complete")




#### WE SAVE THE MODEL SO WE DON'T HAVE TO TRAIN IT AGAIN ####

# Create the directory where all models will be stored if the directory doesn't already exist
os.makedirs('trained_models', exist_ok=True)
# Define the file paths
model_path = f'trained_models/FFDNet_mnist_ensemble_{num_epochs}_epochs_{noise_std_vector}_noise_std_vector.pth'
optimizer_path = f'trained_models/optimizer_FFDNet_mnist_ensemble{num_epochs}_epochs_{noise_std_vector}_noise_std_vector.pth'
# Save the model state dictionary
torch.save(model.state_dict(), model_path)

# Save the optimizer state dictionary
torch.save(optimizer.state_dict(), optimizer_path)

print("Model and optimizer states have been saved.")
