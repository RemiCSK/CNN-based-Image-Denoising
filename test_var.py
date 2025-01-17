######## THIS FILE IS TO TEST THE FFDNet inspired model on the MNIST dataset ########
##### IMPORTANT TO NOTE THAT FOR THE TRAINING OF THIS MODEL WE ALWAYS USED THE SAME NOISE LEVEL MAP#####
### WE WILL VARY THE NOISE LEVEL MAP DURING TRAINING IN AN OTHER MODEL BUT NOT THIS ONE ###


import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms

from every_model_architecture import FFDNet_inspired_small_mnist # Import the FFDNet_inspired_small_mnist model class defined in an other file

# CHECK GPU or choose CPU if no GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



##### DATA PREPARATION #####
# We import both training and testing datasets to evaluate quality on both training and testing data

transform = transforms.Compose([
    transforms.ToTensor()])


batchsize = 10000 # number of imported images
# import training data
try:
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False)

    dataiter_train = iter(trainloader)
    images_train, labels_train = next(dataiter_train)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")


# import testing data
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

# Get the first batch of images and labels
dataiter_test = iter(testloader)
images_test, labels_test = next(dataiter_test)

def add_gaussian_noise_random(tensor, noise_mean, noise_stds):
    """
    Add random Gaussian noise to each image in the dataset.
    Args:
        tensor (torch.Tensor): Input dataset (B, C, H, W).
        noise_mean (float): Mean of the Gaussian noise.
        noise_stds (list of float): List of possible standard deviations for noise.
    Returns:
        torch.Tensor: Dataset with random Gaussian noise added.
    """
    batch_size = tensor.size(0)
    device = tensor.device
    
    # Randomly sample a noise_std for each image in the batch
    random_stds = torch.tensor(noise_stds).to(device)  # Ensure noise_stds is on the same device
    selected_stds = random_stds[torch.randint(0, len(noise_stds), (batch_size,))].view(-1, 1, 1, 1)
    
    # Generate Gaussian noise and add it to the dataset
    noise = torch.randn_like(tensor) * selected_stds + noise_mean
    return tensor + noise

# Define noise parameters


noise_stds = [0.05, 0.1, 0.2, 0.3, 0.5]
noise_std = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# List of possible noise standard deviations
noise_mean = 0.0

# Generate noisy images with random noise levels
noisy_images_train = add_gaussian_noise_random(images_train, noise_mean, noise_stds).to(device)
noisy_images_test = add_gaussian_noise_random(images_test, noise_mean, noise_stds).to(device)

# CHOOSE THE AMOUNT OF NOISE WE WILL TELL THE MODEL TO REMOVE, IT CAN BE DIFFERENT TAN THE AMOUNT OF NOISE ADDED OR THE AMOUNT OF NOISE USED IN TRAINING (In practice for real images we don't know the quantity of noise to denoise that is why we allow for this flexibility)
noise_std_guess = 0.1

# TO CREATE A UNIFORM NOISE LEVEL MAP THAT MATCHES DIMENSIONS OF DOWNSAMPLED IMAGES
def create_uniform_noise_level_map(batch_size, height, width, noise_std_guess):
    """
    Creates a noise level map with random `noise_std_guess` for each image in the batch.
    Args:
        batch_size (int): Number of images in the batch.
        height (int): Height of the images.
        width (int): Width of the images.
        noise_std_guess (torch.Tensor): Tensor of shape (batch_size, 1, 1, 1) with random noise_std values.
    Returns:
        torch.Tensor: Noise level map of shape (batch_size, 1, height//2, width//2).
    """
    noise_level_map = noise_std_guess.expand(-1, 1, height // 2, width // 2)  # Broadcast the noise_std_guess to match the map dimensions
    return noise_level_map

noise_std_range = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.3], device=device)

# Generate random noise_std_guess values for each image in the batch
random_noise_std_guess = noise_std_range[torch.randint(0, len(noise_std_range), (batchsize,), device=device)].view(-1, 1, 1, 1)


noise_std_vector = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] 
##### LOAD THE MODEL #####
num_epochs = 10 # number of epoch of the already trained model
noise_std_train = 0.1 # amount of noise added during the training of the already trained model
model_path = f'trained_models/map_var_ffdnet_inspired_mnist_{num_epochs}_epochs_{noise_std_vector}_noise_std_vector.pth'
optimizer_path = f'trained_models/optimizer_map_var_ffdnet_inspired_mnist{num_epochs}_epochs_{noise_std_vector}_noise_std_vector.pth'

FFDNet_inspired_small_mnist_model = FFDNet_inspired_small_mnist()  # Import class from every_model_architecture.py
print(f"Architecture of small_mnist_model: {FFDNet_inspired_small_mnist_model}")
optimizer = optim.Adam(FFDNet_inspired_small_mnist_model.parameters())

# Load the model state dictionary with map_location to the appropriate device
FFDNet_inspired_small_mnist_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))   #map_location allows to use GPU or CPU even if the model was trained on a GPU

# Load the optimizer state dictionary with map_location to the appropriate device
optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device(device)))


# Move the model to the appropriate device
FFDNet_inspired_small_mnist_model.to(device)

# Set the model to evaluation mode
FFDNet_inspired_small_mnist_model.eval()

print(f"The model : FFDNet_inspired_small_mnist_{num_epochs}_epochs_{noise_std_train}_noise_std and the optimizer states have been loaded.")



##### CHECK THE MODEL ON THE TRAINSET #####

with torch.no_grad():
    b, _, h, w = noisy_images_train.size()
    
    # Generate random noise_std_guess for the batch
    random_noise_std_guess = noise_std_range[torch.randint(0, len(noise_std_range), (b,), device=device)].view(-1, 1, 1, 1)
    
    # Create the noise level map using the random noise_std_guess values
    noise_level_map = create_uniform_noise_level_map(b, h, w, random_noise_std_guess)
    
    # Denoise the images
    denoised_images_train = FFDNet_inspired_small_mnist_model(noisy_images_train, noise_level_map)

# Plot the original, noisy, and denoised images
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i in range(5):
    # Plot original images in the first row
    ax = axes[0, i]
    ax.imshow(images_train[i].cpu().numpy().squeeze(), cmap="gray")
    ax.set_title(f'Original {labels_train[i].item()}')
    ax.axis("off")

    # Plot noisy images in the second row
    ax = axes[1, i]
    ax.imshow(noisy_images_train[i].cpu().numpy().squeeze(), cmap="gray")
    ax.set_title(f'Noisy {labels_train[i].item()}')
    ax.axis("off")

    # Plot denoised images in the third row
    ax = axes[2, i]
    ax.imshow(denoised_images_train[i].cpu().numpy().squeeze(), cmap="gray")
    ax.set_title(f'Denoised {labels_train[i].item()}')
    ax.axis("off")

# Add a title to the entire figure
fig.suptitle(f"Original, Noisy, and Denoised MNIST Images of the train dataset using the model : FFDNet_inspired_small_mnist_{num_epochs}_epochs_{noise_std_train}_noise_std", fontsize=16)
plt.show()








#### Put test results in a function to test with different noise_std_guess parameters
def results_test_presentation(noise_std_vector):
    """
    Takes a predefined vector of noise_std values as input and returns plots and errors of the denoising model on the test set.
    """
    ##### TEST ON THE TEST SET #####
    ##### RUN AND COMPUTE THE LOSS ON THE TEST SET #####
    criterion = nn.MSELoss()

    test_loss = 0.0
    with torch.no_grad():
        b, _, h, w = noisy_images_test.size()  # Get batch size and image dimensions

        # Convert the noise_std_vector to a tensor and move it to the appropriate device
        noise_std_vector_tensor = torch.tensor(noise_std_vector, device=device)

        # Generate random indices to sample from noise_std_vector
        random_indices = torch.randint(0, len(noise_std_vector_tensor), (b,), device=device)

        # Select random noise_std values for the noise level map
        random_noise_std_guess = noise_std_vector_tensor[random_indices].view(-1, 1, 1, 1)

        # Create the noise level map
        noise_level_map = random_noise_std_guess.expand(-1, 1, h // 2, w // 2)

        # Initialize tensor for denoised images
        denoised_images = torch.zeros_like(noisy_images_test).to(device)

        # Start timing
        start_time = time.time()
        
        # Pass noisy images and noise map to the model
        denoised_images = FFDNet_inspired_small_mnist_model(noisy_images_test, noise_level_map)
        
        # End timing
        end_time = time.time()

        # Calculate test loss
        test_loss = criterion(denoised_images, images_test.to(device)).item()
        time_taken = end_time - start_time

    print(f'It took {time_taken} seconds to denoise {noisy_images_test.shape[0]} images from the test dataset and the MSE loss on those images is {test_loss:.4f}')

    # Plot the original, noisy, and denoised images
    fig, axes = plt.subplots(3, 5, figsize=(15, 13))
    for i in range(5):
        # Plot original images in the first row
        ax = axes[0, i]
        ax.imshow(images_test[i].cpu().numpy().squeeze(), cmap="gray")
        ax.set_title(f'Original {labels_test[i].item()}')
        ax.axis("off")

        # Plot noisy images in the second row
        ax = axes[1, i]
        ax.imshow(noisy_images_test[i].cpu().numpy().squeeze(), cmap="gray")
        ax.set_title(f'Noisy {labels_test[i].item()}')
        ax.axis("off")

        # Plot denoised images in the third row
        ax = axes[2, i]
        ax.imshow(denoised_images[i].cpu().numpy().squeeze(), cmap="gray")
        ax.set_title(f'Denoised {labels_test[i].item()}')
        ax.axis("off")

    # Add a title to the entire figure
    fig.suptitle(f"Original, Noisy, and Denoised MNIST Images of the test dataset\n"
                 f"Using the model: FFDNet_inspired_small_mnist_{num_epochs}_epochs with noise_std_vector\n"
                 f"Random noise_std values from {noise_std_vector} used for the noise level map.\n"
                 f"It took {time_taken} seconds to denoise {noisy_images_test.shape[0]} images and the MSE loss on those images is {test_loss:.4f}", fontsize=16)
    plt.show()

    # Reset tensors to free memory
    denoised_images = None
    noise_level_map = None
    return


guess = [0.1, 0.2, 0.3, 0.4, 0.5]

#reminder, true noise std is 0.1
print("DONT LOOK AT THE PLOT JUST BELOW, IT'S JUST A FIRST RUN TO INITIALIZE TENSORS")
poubelle = results_test_presentation(guess) # Note that the first time we call for our function, the time will be longer because of initialization of various tensors. That is why we make a first run that we will not use

print("NOW YOU CAN LOOK AT THE PLOT")

results_test_presentation(guess) # we do a SUPER underestimation of the noise added to the images

print("We see that underestimating or matching the noise level map gives good results. However, overestimating the noise level map gives poor results. We verified what the authors of FFDNet said in their article.")