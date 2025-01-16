import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms

from every_model_architecture import Small_MNIST  # Import the Small_MNIST model class defined in an other file

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

def add_gaussian_noise(tensor, noise_mean, noise_std):
    #noise = torch.randn(tensor.size(),device=tensor.device) * noise_std + noise_mean
    noise = torch.randn(tensor.size()) * noise_std + noise_mean
    return tensor + noise


# CHOOSE THE AMOUNT OF NOISE TO BE ADDED THEN REMOVED, IT CAN BE SOMETHING DIFFERENT THAN THE PARAMETERS USED FOR TRAINING
# Note that for the training I used noise_std = 1 that is ridiculously high (almost only noise). That is why results are not good for the moment.
noise_mean = 0
noise_std = 0.1

# Add Gaussian noise to the images
noisy_images_train = add_gaussian_noise(images_train, noise_mean,noise_std).to(device)
noisy_images_test = add_gaussian_noise(images_test, noise_mean,noise_std).to(device)




##### LOAD THE MODEL #####
epoch = 30 # number of epoch of the already trained model
noise_std_train = 0.1 # amount of noise added during the training of the already trained model
model_path = f'trained_models/small_mnist_model_{epoch}_epochs_{noise_std_train}_noise_std.pth'
optimizer_path = f'trained_models/optimizer_mnist_model_{epoch}_epochs_{noise_std_train}_noise_std.pth'

small_mnist_model = Small_MNIST()  # Import class from every_model_architecture.py
print(f"Architecture of small_mnist_model: {small_mnist_model}")
optimizer = optim.Adam(small_mnist_model.parameters())

# Load the model state dictionary with map_location to the appropriate device
small_mnist_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))   #map_location allows to use GPU or CPU even if the model was trained on a GPU

# Load the optimizer state dictionary with map_location to the appropriate device
optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device(device)))


# Move the model to the appropriate device
small_mnist_model.to(device)

# Set the model to evaluation mode
small_mnist_model.eval()

print("The small_mnist_model and optimizer states have been loaded.")



##### CHECK THE MODEL ON THE TRAINSET #####

with torch.no_grad():
    denoised_images_train = small_mnist_model(noisy_images_train)

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
fig.suptitle("Original, Noisy, and Denoised MNIST Images of the train dataset", fontsize=16)
plt.show()





##### TEST ON THE TEST SET #####

# Run the model on the noisy images
with torch.no_grad():
    denoised_images = small_mnist_model(noisy_images_test)

# Plot the original, noisy, and denoised images
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
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
fig.suptitle("Original, Noisy, and Denoised MNIST Images of the test dataset", fontsize=16)
plt.show()



##### COMPUTE THE LOSS ON THE TEST SET #####
criterion = nn.MSELoss()

test_loss = 0.0
with torch.no_grad():
    start_time = time.time()
    denoised_images_test = small_mnist_model(noisy_images_test)
    end_time = time.time()
    test_loss = criterion(denoised_images_test, images_test.to(device)).item()
    time_taken = end_time - start_time
print(f'It took {time_taken} seconds to denoise {batchsize} images from the test dataset and the MSE loss on those images is {test_loss:.4f}')


print(noisy_images_test.shape)