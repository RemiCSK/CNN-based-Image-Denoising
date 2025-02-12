import torch.nn as nn
import torch
###### PUT ARCHITECTURE OF EACH MODEL HERE AS A CLASS ######


###### Define a class for the small_mnist_model #####   # we use classes to export and import the model easily
# Smaller model for MMNIST dataset, this is the same model as big_model but with less intermediate convolution layers
# THis model DOES NOT take a noise map as input. It just down samples, do convolutions and reverse the down sample operation.

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


###### Define a class for FFDNet_inspired_small_mnist model #####   # we use classes to export and import the model easily
# This is a very similar model to Small_mnist_model but we add a noise map as input, inspired by what is done in FFDNet


class FFDNet_inspired_small_mnist(nn.Module):

    def __init__(self):
        super(FFDNet_inspired_small_mnist, self).__init__()
        self.model = nn.Sequential(
            # REVERSIBLE DOWN SAMPLING is done in the forward function to apply it only to images and not the noise map
            # First convolution layer
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1),   # 4+1=5 input channels because we will add a noise map
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
    def forward(self, x, noise_level_map):
        downsampled_images = nn.PixelUnshuffle(2)(x) #reversible down sample images in 4 sub images
        #### CHECKING DIMENSION OF THE NOISE LEVEL MAP ####
        batch_size, _, h, w = downsampled_images.size()
        noise_level_map = noise_level_map.view(batch_size, 1, h, w)   # reshape the noise level map to match dimension of down sample images if necessary
        concat_input = torch.cat([downsampled_images, noise_level_map], dim=1) # concatenates the 4 downsampled images and the noise level map
        return self.model(concat_input)



# Create a very similar model to the one just above but we simply add a convolution layer

class FFDNet_inspired_small_mnist_extend(nn.Module):

    def __init__(self):
        super(FFDNet_inspired_small_mnist_extend, self).__init__()
        self.model = nn.Sequential(
            # REVERSIBLE DOWN SAMPLING is done in the forward function to apply it only to images and not the noise map
            # First convolution layer
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1),   # 4+1=5 input channels because we will add a noise map
            nn.ReLU(inplace=True),
            # 2 convolution layers with batch normalization and ReLU
            *[nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)) for _ in range(3)],
            # Final convolution layer
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, padding=1),
            # ADD THE REVERSE OF THE DOWN SAMPLING
            nn.PixelShuffle(2)
)
    def forward(self, x, noise_level_map):
        downsampled_images = nn.PixelUnshuffle(2)(x) #reversible down sample images in 4 sub images
        #### CHECKING DIMENSION OF THE NOISE LEVEL MAP ####
        batch_size, _, h, w = downsampled_images.size()
        noise_level_map = noise_level_map.view(batch_size, 1, h, w)   # reshape the noise level map to match dimension of down sample images if necessary
        concat_input = torch.cat([downsampled_images, noise_level_map], dim=1) # concatenates the 4 downsampled images and the noise level map
        return self.model(concat_input)


# Create a very similar model to the one just above but we use a single intermediate convolution layer

class FFDNet_inspired_small_mnist_extend2(nn.Module):

    def __init__(self):
        super(FFDNet_inspired_small_mnist_extend2, self).__init__()
        self.model = nn.Sequential(
            # REVERSIBLE DOWN SAMPLING is done in the forward function to apply it only to images and not the noise map
            # First convolution layer
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1),   # 4+1=5 input channels because we will add a noise map
            nn.ReLU(inplace=True),
            # 2 convolution layers with batch normalization and ReLU
            *[nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)) for _ in range(1)],
            # Final convolution layer
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, padding=1),
            # ADD THE REVERSE OF THE DOWN SAMPLING
            nn.PixelShuffle(2)
)
    def forward(self, x, noise_level_map):
        downsampled_images = nn.PixelUnshuffle(2)(x) #reversible down sample images in 4 sub images
        #### CHECKING DIMENSION OF THE NOISE LEVEL MAP ####
        batch_size, _, h, w = downsampled_images.size()
        noise_level_map = noise_level_map.view(batch_size, 1, h, w)   # reshape the noise level map to match dimension of down sample images if necessary
        concat_input = torch.cat([downsampled_images, noise_level_map], dim=1) # concatenates the 4 downsampled images and the noise level map
        return self.model(concat_input)





class FFDNet_inspired_small_cifar(nn.Module):

    def __init__(self):
        super(FFDNet_inspired_small_cifar, self).__init__()
        self.model = nn.Sequential(
            # First convolution layer
            nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1),  # 12 (downsampled RGB) + 1 (noise map)
            nn.ReLU(inplace=True),
            # 2 convolution layers with batch normalization and ReLU
            *[nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)) for _ in range(2)],
            # Final convolution layer
            nn.Conv2d(in_channels=64, out_channels=12, kernel_size=3, padding=1),
            # Reversible up-sampling
            nn.PixelShuffle(2)
        )

    def forward(self, x, noise_level_map):
        # Reversible down-sampling (PixelUnshuffle)
        downsampled_images = nn.PixelUnshuffle(2)(x)  # Input shape (B, 3, H, W) -> Output shape (B, 12, H/2, W/2)

        # Adjust the noise map dimensions to match downsampled images
        batch_size, _, h, w = downsampled_images.size()
        noise_level_map = noise_level_map.view(batch_size, 1, h, w)  # Move noise_level_map to the same device as x

        # Concatenate downsampled images and noise map
        concat_input = torch.cat([downsampled_images, noise_level_map], dim=1)  # Shape (B, 13, H/2, W/2)

        return self.model(concat_input)





#### Define an ensemble model that takes the average of the output of two models ####

class FFDNet_mnist_ensemble(nn.Module):
    def __init__(self, model1, model2):
        super(FFDNet_mnist_ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x, noise_level_map):
        return 0.5 * (self.model1(x, noise_level_map) + self.model2(x, noise_level_map))  # Average the output of the two models


# Defiine a class to combine the output of two models with a small neural network instead of averaging
class Neural_combining(nn.Module):
    def __init__(self):
        super(Neural_combining, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)





# Define a class for an other ensemble model. But the combination is done via a small neural network.
class FFDNet_mnist_ensemble_neural(nn.Module):
    def __init__(self, model1, model2):
        super(FFDNet_mnist_ensemble_neural, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.neural_combining = Neural_combining()


    def forward(self, x, noise_level_map):
        output1 = self.model1(x, noise_level_map)
        output2 = self.model2(x, noise_level_map)
        combined_output = torch.cat([output1, output2], dim=1)  # CONCATENATE output of both models
        return self.neural_combining(combined_output)
