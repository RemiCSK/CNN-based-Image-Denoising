from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from every_model_architecture import FFDNet_inspired_small_mnist  # Import the model class

# Define the path to the JPEG file
jpeg_file_path = 'data/real_data/real_real_noise.jpeg'  # Update with the correct path to your JPEG file

# Load the image
image = Image.open(jpeg_file_path).convert('L')  # Convert to grayscale ('L' mode)

# Define the transform to convert the image to a tensor
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to the same size as CIFAR images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to the same range as CIFAR images
])

# Apply the transform to the image
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Ensure the tensor has the same type as CIFAR images
image_tensor = image_tensor.type(torch.FloatTensor)

# Print the tensor shape and type to verify
print(f"Image tensor shape: {image_tensor.shape}")
print(f"Image tensor type: {image_tensor.dtype}")

# Load the model
model = FFDNet_inspired_small_mnist()
# Update with the correct path to your model weights
model_path = f'trained_models/FFDNet_inspired_small_mnist_{100}_epochs_{0.1}_noise_std.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Create a noise level map
noise_std = 0.001  # Example noise standard deviation
noise_level_map = torch.full((1, 1, 16, 16), noise_std)  # Adjust dimensions as necessary

# Denoise the image
with torch.no_grad():
    denoised_image_tensor = model(image_tensor, noise_level_map)

# Plot the original and denoised images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot original image
axes[0].imshow(image_tensor.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Plot denoised image
axes[1].imshow(denoised_image_tensor.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title('Denoised Image')
axes[1].axis('off')

plt.show()