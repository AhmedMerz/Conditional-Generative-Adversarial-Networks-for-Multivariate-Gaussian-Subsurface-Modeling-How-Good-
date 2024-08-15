import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
z_dim = 100
gen_hidden_dim = 128
critic_hidden_dim = 128
lr = 0.0002
batch_size = 20
num_epochs = 950
critic_iterations = 5
clip_value = 0.01  # Clip value for critic weights


import matplotlib.pyplot as plt

def save_or_show_image(img_tensor, filename=None, show=False):
    """
    Converts a PyTorch tensor into an image and saves or displays it.
    
    Args:
    - img_tensor (torch.Tensor): The image tensor to convert.
    - filename (str): The filename to save the image to.
    - show (bool): Whether to display the image rather than saving.
    """
    # Remove the batch dimension and the normalization
    img_tensor = img_tensor[0].detach().cpu()  # Assuming img_tensor is 4D (BxCxHxW) with B=1
    img_tensor = (img_tensor + 1) / 2  # Undo the normalization
    img = transforms.ToPILImage()(img_tensor)
    
    if show:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        img.save(filename)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            # Project and reshape
            nn.Linear(z_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),  # Becomes (batch_size, 256, 8, 8)

            # Upsample to (batch_size, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Upsample to (batch_size, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Upsample to (batch_size, 32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Upsample to (batch_size, 1, 128, 128)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output is (batch_size, 1, 128, 128)
        )

    def forward(self, z):
        return self.net(z)



class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Convolution Block 1
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Convolution Block 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Convolution Block 3
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten and Output
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1)
        )

    def forward(self, img):
        x = self.net(img)
        return x



def scale_to_original(generated_images, min_val, max_val, mean, std):
    # Reverse the normalization
    images = generated_images * std + mean
    # Scale back to original range
    images = images * (max_val - min_val) + min_val
    return images


# Initialize generator and critic
img_shape = (1, 128, 128)
generator = Generator(z_dim).to(device)
critic = Critic().to(device)

# Optimizers
optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = optim.RMSprop(critic.parameters(), lr=lr)


# Define transformations for grayscale images
transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))  # This assumes the data is already scaled to [0, 1]
])

# Define a custom transform to scale the pixel values to [0, 1]
class ScaleToUnitInterval(object):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        return (sample - self.min_val) / (self.max_val - self.min_val)
# Update the GrayscaleNPZDataset to apply the ScaleToUnitInterval transform before ToTensor
class GrayscaleNPZDataset(Dataset):
    """Custom Dataset for loading grayscale images from an npz file"""
    
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['my_array1']
        self.scaler = ScaleToUnitInterval(np.min(self.images), np.max(self.images))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get the image at the specified index
        image = self.images[idx]
        
        # Scale the image pixel values to [0, 1]
        image = self.scaler(image)
        
        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        
        # Add a channel dimension (PyTorch expects CxHxW)
        image = image.unsqueeze(0)
        
        # Apply additional transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image

# Create the dataset with the defined transformations
dataset = GrayscaleNPZDataset('C:/Users/am224745/Documents/Codes AM/SGS/my_data100.npz', transform=transform)

def scale_to_original(generated_images, original_min=2.75e-5, original_max=0.219986):
    # Reverse the normalization (assuming the original normalization was (value - 0.5) / 0.5)
    images = (generated_images + 1) / 2  # Undo the [-1, 1] normalization to get [0, 1]

    # Scale back to original range
    images = images * (original_max - original_min) + original_min
    return images



original_min, original_max = 0, 0.219986

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
os.chdir('C:/Users/am224745/Documents/Codes AM/SGS/Gaussian')
# Check the shape of the first batch of images
first_batch_images = next(iter(dataloader))
first_batch_images.shape
# 
# Create directories if they do not exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if not os.path.exists('generated_images2'):
    os.makedirs('generated_images2')
# Training
for epoch in range(num_epochs):
    for i, imgs in enumerate(dataloader):

        # Train Critic
        for _ in range(critic_iterations):
            critic.zero_grad()
            real_imgs = imgs.to(device).float()
            real_imgs = real_imgs.view(real_imgs.size(0), 1, 128, 128)  # Ensure the correct shape for the Critic
            z = torch.randn(imgs.shape[0], z_dim).to(device).float()
            fake_imgs = generator(z).detach()
            loss_critic = -(torch.mean(critic(real_imgs)) - torch.mean(critic(fake_imgs)))
            loss_critic.backward()
            optimizer_D.step()
            
            # Clip weights of critic
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # Train Generator
        generator.zero_grad()
        gen_imgs = generator(z)
        loss_gen = -torch.mean(critic(gen_imgs))
        loss_gen.backward()
        optimizer_G.step()
        print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
              Loss D: {loss_critic.item()}, loss G: {loss_gen.item()}")
              
        '''
    if (epoch+1) % 100 == 0:
        with torch.no_grad():
            generator.eval()  # Set the generator to evaluation mode
            z = torch.randn(10, z_dim).to(device)  # Generate random noise
            fake_images = generator(z)  # Generate fake images
        
            # Save generated images
            for j, fake_img in enumerate(fake_images):
                save_or_show_image(fake_img.unsqueeze(0), filename=f'50/epoch_{epoch+1}_image_{j}.png', show=False)
            
            generator.train()  # Set the generator back to training mode
            '''
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            generator.eval()  # Set the generator to evaluation mode
            z = torch.randn(100, z_dim).to(device)  # Generate random noise
            fake_images = generator(z)  # Generate fake images
            fake_images = scale_to_original(fake_images)
                    # Convert images to numpy arrays after moving them to CPU
            np_images = [img.cpu().numpy() for img in fake_images]
        
                    # Save the numpy arrays in an NPZ file
            np.savez(f'100/epoch_{epoch+1}.npz', *np_images)
        
            generator.train()
        
        # Save model checkpoints
        torch.save(generator.state_dict(), f'100/checkpoints/generator_epoch_{epoch+1}.pth')
        torch.save(critic.state_dict(), f'100/checkpoints/critic_epoch_{epoch+1}.pth')
        

for layer in generator.modules():
    layer_type = str(layer.__class__).split('.')[-1].split("'")[0]
    output_shape = "To calculate"
    kernel_size = getattr(layer, 'kernel_size', None)
    stride = getattr(layer, 'stride', None)
    padding = getattr(layer, 'padding', None)
    params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

    # Add a row to your table here with these values


def get_layer_details(model):
    layers = []
    for layer in model.modules():
        if len(list(layer.children())) > 0:
            # Skip layers which are just container of other layers
            continue

        layer_type = layer.__class__.__name__
        params = sum(p.numel() for p in layer.parameters())
        
        # For convolutional layers
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
        else:
            kernel_size = stride = padding = None

        # For linear layers
        if isinstance(layer, nn.Linear):
            output_shape = (layer.out_features,)
        # Add other layer types if needed
        else:
            output_shape = "Varies"

        layer_info = {
            "Type": layer_type,
            "Output Shape": output_shape,
            "Kernel Size": kernel_size,
            "Stride": stride,
            "Padding": padding,
            "Parameters": params
        }
        layers.append(layer_info)
    
    return layers

# Now use this function for your Generator and Critic
generator_details = get_layer_details(generator)
critic_details = get_layer_details(critic)

# You can now print these details or convert them to a table format as needed
for layer in generator_details:
    print(layer)

for layer in critic_details:
    print(layer)


# Save the generator's state_dict
#torch.save(generator.state_dict(), 'wgan_generator800.pth')
'''


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Initialize generator and critic
generator = Generator(z_dim).to(device)
critic = Critic().to(device)

# Count parameters
generator_params = count_parameters(generator)
critic_params = count_parameters(critic)

print("Generator Parameters:", generator_params)
print("Critic Parameters:", critic_params)
print("Total Parameters in GAN:", generator_params + critic_params)


generator.load_state_dict(torch.load('wgan_generator2.pth'))
generator.to(device)
generator.eval()  # Set the generator to evaluation mode

# Create directory for generated images if it does not exist
generated_images_dir = 'generated_images3'
os.makedirs(generated_images_dir, exist_ok=True)

# Generate and save 500 images
with torch.no_grad():
    for i in range(500):
        z = torch.randn(1, z_dim).to(device)  # Generate random noise vector
        fake_image = generator(z)  # Generate fake image
        filename = os.path.join(generated_images_dir, f'generated_image_{i:04d}.png')
        save_or_show_image(fake_image, filename=filename, show=False)
        
  '''