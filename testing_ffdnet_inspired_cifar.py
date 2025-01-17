######## THIS FILE IS TO TEST THE FFDNet inspired model on the cifar dataset ########
##### IMPORTANT TO NOTE THAT FOR THE TRAINING OF THIS MODEL WE ALWAYS USED THE SAME NOISE LEVEL MAP#####
### WE WILL VARY THE NOISE LEVEL MAP DURING TRAINING IN AN OTHER MODEL BUT NOT THIS ONE ###


import matplotlib.pyplot as plt
import torch
from torchvision.datasets import cifar
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms

from every_model_architecture import FFDNet_inspired_small_cifar # Import the FFDNet_inspired_small_cifar model class defined in an other file

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
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False)

    dataiter_train = iter(trainloader)
    images_train, labels_train = next(dataiter_train)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")


# import testing data
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

# Get the first batch of images and labels
dataiter_test = iter(testloader)
images_test, labels_test = next(dataiter_test)

def add_gaussian_noise(tensor, noise_mean, noise_std):
    #noise = torch.randn(tensor.size(),device=tensor.device) * noise_std + noise_mean
    noise = torch.randn(tensor.size()) * noise_std + noise_mean
    return tensor + noise


# CHOOSE THE AMOUNT OF NOISE TO BE ADDED, IT CAN BE SOMETHING DIFFERENT THAN THE PARAMETERS USED FOR TRAINING
# Note that for training I used std = 0.1
noise_std = 0.1
noise_mean = 0.0


# Add Gaussian noise to the images
noisy_images_train = add_gaussian_noise(images_train, noise_mean,noise_std).to(device)
noisy_images_test = add_gaussian_noise(images_test, noise_mean,noise_std).to(device)


# CHOOSE THE AMOUNT OF NOISE WE WILL TELL THE MODEL TO REMOVE, IT CAN BE DIFFERENT TAN THE AMOUNT OF NOISE ADDED OR THE AMOUNT OF NOISE USED IN TRAINING (In practice for real images we don't know the quantity of noise to denoise that is why we allow for this flexibility)
noise_std_guess = 0.1

# TO CREATE A UNIFORM NOISE LEVEL MAP THAT MATCHES DIMENSIONS OF DOWNSAMPLED IMAGES
def create_uniform_noise_level_map(batch_size, height, width, noise_std):
    noise_level_map = torch.full((batch_size, 1, height // 2, width // 2), noise_std).to(device) # Creates noise level map with only noise_std values in it.
    return noise_level_map



##### LOAD THE MODEL #####
num_epochs = 10 # number of epoch of the already trained model
noise_std_train = 0.1 # amount of noise added during the training of the already trained model
model_path = f'trained_models/FFDNet_inspired_small_cifar_{num_epochs}_epochs_{noise_std_train}_noise_std.pth'
optimizer_path = f'trained_models/optimizer_FFDNet_inspired_small_cifar{num_epochs}_epochs_{noise_std_train}_noise_std.pth'

FFDNet_inspired_small_cifar_model = FFDNet_inspired_small_cifar()  # Import class from every_model_architecture.py
print(f"Architecture of small_cifar_model: {FFDNet_inspired_small_cifar_model}")
optimizer = optim.Adam(FFDNet_inspired_small_cifar_model.parameters())

# Load the model state dictionary with map_location to the appropriate device
FFDNet_inspired_small_cifar_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))   #map_location allows to use GPU or CPU even if the model was trained on a GPU

# Load the optimizer state dictionary with map_location to the appropriate device
optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device(device)))


# Move the model to the appropriate device
FFDNet_inspired_small_cifar_model.to(device)

# Set the model to evaluation mode
FFDNet_inspired_small_cifar_model.eval()

print(f"The model : FFDNet_inspired_small_cifar_{num_epochs}_epochs_{noise_std_train}_noise_std and the optimizer states have been loaded.")



##### CHECK THE MODEL ON THE TRAINSET #####

with torch.no_grad():
    b, _, h, w = noisy_images_train.size()   # to create noise level map and making sure dimensions match
    noise_level_map = create_uniform_noise_level_map(b, h, w, noise_std_guess)  #noise_std_guess because that's a guess of noise we think the images have. Then it creates a noise_level_map that we will give to the model.

    denoised_images_train = FFDNet_inspired_small_cifar_model(noisy_images_train, noise_level_map)

# Plot the original, noisy, and denoised images
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i in range(5):
    # Plot original images in the first row
    ax = axes[0, i]
    ax.imshow(images_train[i].cpu().numpy().transpose(1, 2, 0))  # Transpose to (H, W, C)
    ax.set_title(f'Original {labels_train[i].item()}')
    ax.axis("off")

    # Plot noisy images in the second row
    ax = axes[1, i]
    ax.imshow(noisy_images_train[i].cpu().numpy().transpose(1, 2, 0))
    ax.set_title(f'Noisy {labels_train[i].item()}')
    ax.axis("off")

    # Plot denoised images in the third row
    ax = axes[2, i]
    ax.imshow(denoised_images_train[i].cpu().numpy().transpose(1, 2, 0))
    ax.set_title(f'Denoised {labels_train[i].item()}')
    ax.axis("off")

# Add a title to the entire figure
fig.suptitle(f"Original, Noisy, and Denoised cifar Images of the train dataset using the model : FFDNet_inspired_small_cifar_{num_epochs}_epochs_{noise_std_train}_noise_std", fontsize=16)
plt.show()








#### Put test results in a function to test with different noise_std_guess parameters
def results_test_presentation(noise_std_guess):
    """
    Takes noise_std_guess as input and returns plots and errors of the denoising model on the test set.
    """
    ##### TEST ON THE TEST SET #####
        ##### RUN AND COMPUTE THE LOSS ON THE TEST SET #####
    criterion = nn.MSELoss()

    test_loss = 0.0
    with torch.no_grad():
        b, _, h, w = noisy_images_train.size()   # to create noise level map and making sure dimensions match
        noise_level_map = create_uniform_noise_level_map(b, h, w, noise_std_guess)  #noise_std_guess because that's a guess of noise we think the images have. Then it creates a noise_level_map that we will give to the model.

        denoised_images = torch.zeros_like(noisy_images_test).to(device) # initialize tensor so it doesn't count in the time to denoise
        start_time = time.time()   # Note that the first time we call for our function, the time will be longer because of initialization of various tensors
        denoised_images = FFDNet_inspired_small_cifar_model(noisy_images_test,noise_level_map)
        end_time = time.time()
        test_loss = criterion(denoised_images, images_test.to(device)).item()
        time_taken = end_time - start_time
    print(f'It took {time_taken} seconds to denoise {noisy_images_test.shape[0]} images from the test dataset and the MSE loss on those images is {test_loss:.4f}')


    # Plot the original, noisy, and denoised images
    fig, axes = plt.subplots(3, 5, figsize=(15, 13))
    for i in range(5):
        # Plot original images in the first row
        ax = axes[0, i]
        ax.imshow(images_test[i].cpu().numpy().transpose(1, 2, 0))
        ax.set_title(f'Original {labels_test[i].item()}')
        ax.axis("off")

        # Plot noisy images in the second row
        ax = axes[1, i]
        ax.imshow(noisy_images_test[i].cpu().numpy().transpose(1, 2, 0))
        ax.set_title(f'Noisy {labels_test[i].item()}')
        ax.axis("off")

        # Plot denoised images in the third row
        ax = axes[2, i]
        ax.imshow(denoised_images[i].cpu().numpy().transpose(1, 2, 0))
        ax.set_title(f'Denoised {labels_test[i].item()}')
        ax.axis("off")

    # Add a title to the entire figure
    fig.suptitle(f"Original, Noisy, and Denoised cifar Images of the test dataset\n"
             f"Using the model: FFDNet_inspired_small_cifar_{num_epochs}_epochs_{noise_std}_noise_std\n"
             f"True noise std added during training: {noise_std_train}\n"
             f"True noise std added to the clean images for testing is {noise_std}\n"
             f"Additional input of the model is a uniform noise level map with value {noise_std_guess}.\n"
             f"It took {time_taken} seconds to denoise {noisy_images_test.shape[0]} images and the MSE loss on those images is {test_loss:.4f}", fontsize=16)
    plt.show()

    # Reset the denoised_images and noise_level_map tensors otherwise, the first time we run this function it will be slower. This makes time measurements more consistent to make comparisons.
    # Note that the first time we call for our function, the time will be longer because of initialization of various tensors
    denoised_images = None
    noise_level_map = None
    return




#reminder, true noise std is 0.1
print("DONT LOOK AT THE PLOT JUST BELOW, IT'S JUST A FIRST RUN TO INITIALIZE TENSORS")
poubelle = results_test_presentation(1) # Note that the first time we call for our function, the time will be longer because of initialization of various tensors. That is why we make a first run that we will not use

print("NOW YOU CAN LOOK AT THE FOLLOWING PLOTS")
print("Reminder, true noise std is 0.1")
results_test_presentation(0.01) # we do a SUPER underestimation of the noise added to the images
results_test_presentation(0.08) # we do an underestimation of the noise added to the images
results_test_presentation(0.1) # noise_std_guess = 0.1 (matches the noised used for training)
results_test_presentation(0.5) # we do an overstimation of the noise added to the images
results_test_presentation(1) # we do a SUPER overstimation of the noise added to the images

print("We see that underestimating or matching the noise level map gives good results. However, overestimating the noise level map gives poor results. We verified what the authors of FFDNet said in their article.")
