# Import Libraries
import itertools
import os
import json
import torch
import torchvision
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from resnet_vae import ResNet_VAE
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Argument Parser
parser = argparse.ArgumentParser(description='Convolutional AutoEncoder for Geometry Data')
parser.add_argument('--device', type=int,nargs='+', default=[0], help='Device to run the model on')
parser.add_argument('--lrs', type=float, nargs='+', default=[0.0005, 0.001, 0.005], help='Learning Rates to try')
parser.add_argument('--bs', type=int, nargs='+', default=[32], help='Batch Sizes to try')
parser.add_argument('--embed_sizes', type=int, nargs='+', default=[512,1024], help='Embedding Sizes to try')
args = parser.parse_args()
def loss_function(reconstruction_loss, x, x_hat, mu, logvar, kl_factor = 0.01):
    recon_loss = reconstruction_loss(x, x_hat)
    kl_loss = 0.5 * torch.mean(-1 - logvar + mu ** 2 + logvar.exp())
    # print(f'KL Loss: {kl_loss} Reconstruction Loss: {recon_loss}')
    return kl_factor*kl_loss + recon_loss

# Load Config files
path = os.getcwd()
config_path = os.path.join(path, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

print("The Configuration Variables are:")
print('Configuration: ', config)

# Define Config variables
image_size = config['image_size']
data_path = config['DataPath']
batch_size = config['batch_size']
learning_rate = config['lr']
weight_decay = config['weight_decay']
epochs = config['n_epochs']
load_model = config['load_model']
embed_size = config['embedding_size']
embed_sizes = args.embed_sizes
lrs = args.lrs
bs_sizes = args.bs
print("\n____________________________________________________\n")
print("\nLoading Dataset into DataLoader...")

all_imgs = glob.glob(data_path + '/*.npy')
shuffle(all_imgs)

# Train Images
split_ratio = 0.9
split_index = int(len(all_imgs) * split_ratio)
train_imgs = all_imgs[:split_index]
test_imgs = all_imgs[split_index:]



# DataLoader Function
class imagePrep(torch.utils.data.Dataset):
    def __init__(self, images, transform,rotate=False,flip_horizontal=False,flip_vertical=False):
        super().__init__()
        self.paths = images
        self.len = len(self.paths)
        self.transform = transform
        self.rotate = rotate
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = (255*np.load(path)[:, 0, :].T).astype(np.uint8)
        if self.rotate:
            image = image.T
        if self.flip_horizontal:
            image = np.flip(image, axis=1)
        if self.flip_vertical:
            image = np.flip(image, axis=0)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image


# Dataset Transformation Function
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                    torchvision.transforms.Resize((image_size[0], image_size[1])),
                                                    torchvision.transforms.ToTensor()])

# Apply Transformations to Data
train_set = imagePrep(train_imgs, dataset_transform)
train_set_rotate = imagePrep(train_imgs, dataset_transform, rotate=True)
train_set_flip_horizontal =imagePrep(train_imgs, dataset_transform, flip_horizontal=True)
train_set_flip_vertical = imagePrep(train_imgs, dataset_transform, flip_vertical=True)
train_set_flip_both = imagePrep(train_imgs, dataset_transform, flip_horizontal=True, flip_vertical=True)
train_set = torch.utils.data.ConcatDataset([train_set, train_set_rotate, train_set_flip_horizontal, train_set_flip_vertical, train_set_flip_both])

print("\nDataLoader Set!")
print("\n____________________________________________________\n")

print("\nBuilding Convolutional AutoEncoder Network Model...")


print("\n____________________________________________________\n")


model_loss_dicts = {}
model_weights_dicts = {}
for learning_rate, batch_size, embed_size in itertools.product(lrs, bs_sizes, embed_sizes):
    print(f"Learning Rate: {learning_rate} | Batch Size: {batch_size} | Embedding Size: {embed_size}")
    model_name = f'geometry_convVAE_model_{learning_rate}_{batch_size}_{embed_size}.pth'
    print("\n____________________________________________________\n")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(imagePrep(test_imgs, dataset_transform), batch_size=batch_size)
    # defining the device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device[0]}')
    else:
        device = torch.device("cpu")
    convAE_model = torch.nn.DataParallel(ResNet_VAE(CNN_embed_dim=embed_size), device_ids=args.device).to(device)
    if load_model:
        convAE_model.load_state_dict(torch.load('geometry_convVAE_model.pth', map_location=device))
    optimizer = torch.optim.Adam(convAE_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    recon_loss = torch.nn.BCELoss().to(device)


    print("\nTraining the Convolutional AutoEncoder Model on Training Data...")

    # Training of Model

    losses = []
    best_loss = 999999
    for epoch in range(epochs):
        epoch_loss = 0
        for X in train_loader:
            img = X.to(device)
            img = torch.autograd.Variable(img)

            recon, mu, logvar, latents = convAE_model(img, latent_vec=True)
            # plt.imshow(recon.cpu().data[1][0], cmap='gray')
            # plt.title('Reconstructed Image')
            # plt.figure()
            # plt.imshow(img.cpu().data[1][0], cmap='gray')
            # plt.title('Original Image')
            # plt.show()
            loss = loss_function(recon_loss, recon, img, mu, logvar)

            # Backward Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = convAE_model.state_dict()
        print("\nEpoch: {} | Loss: {:.4f}".format(epoch + 1, epoch_loss), ' | Best Loss: {:.4f}'.format(best_loss))
    model_loss_dicts[model_name] = best_loss.item()
    model_weights_dicts[model_name] = best_weights
    torch.save(best_weights, model_name)
    print("\n____________________________________________________\n")
print(model_loss_dicts)
best_model = min(model_loss_dicts, key=model_loss_dicts.get)
print(f"Best Model: {best_model} | Best Loss: {model_loss_dicts[best_model]}")


