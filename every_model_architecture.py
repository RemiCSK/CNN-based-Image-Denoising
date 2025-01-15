import torch.nn as nn

###### PUT ARCHITECTURE OF EACH MODEL HERE AS A CLASS ######


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
