"""Create a .tsv file containing predictions of an image dataset."""

import os
import shutil
import random
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.nn.functional as F
import math
import cnn_models
import cnn
from constants import *
import pandas as pd

class myDataset(torch.utils.data.Dataset):
    """Enable creating train, test, and dev test datasets for PyTorch."""

    def __init__(self, root_dir=TEST_IMAGES_DIR, transform=None):
        """Define the transformation and imagepaths for the dataset."""
        self.transform = transform
        self.root_dir = root_dir
        self.images = [root_dir + img for img in os.listdir(root_dir)]

    def __len__(self):
        """Calculate the number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get the image."""
        img_path = self.images[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img


def main(model: nn.Module):
    """Start the CNN model."""
    if IMAGENET:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.42986962, 0.42986962, 0.42986962], [0.2697042, 0.2697042, 0.2697042]
    pred_loader = transform_data(mean, std)
    model.load_state_dict(torch.load(PATH))
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    create_predictions(model, device, pred_loader)


# def transform_data(mean: float, std: float, dict_labels):
def transform_data(mean: float, std: float):
    """Transform images to usable matrix representation."""
    transform = transforms.Compose([
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                        mean=mean,
                                        std=std
                                        )])
                                
    pred_set = myDataset(root_dir=TEST_IMAGES_DIR, transform=transform)
    pred_loader = torch.utils.data.DataLoader(dataset=pred_set, batch_size=100, shuffle=False)
    return pred_loader


def create_predictions(model: nn.Module, device, pred_loader):
    """Create a .tsv file for prections based on a specified model."""
    pred_file = DATA_DIR + 'pred.tsv'
    model.eval()
    image_names = [img[2:-4] for img in TEST_IMAGE_FILE_NAMES]
    image_names.sort()  # because os.listdir documentation says it can be in arbitary order, for dataloader this is not arbitary
    pred_array = []
    with torch.no_grad():
        for _, data in enumerate(pred_loader):
            data = data.to(device)
            pred = model(data)
            pred = torch.round(torch.sigmoid(pred))
            if len(pred_array) == 0:
                pred_array = pred.numpy()
            else:
                pred_array = np.concatenate((pred_array, pred.numpy()))
    df = pd.DataFrame(pred_array, columns=ANNOTATIONS)
    df.insert(0, 'Filename', image_names)
    df.to_csv(pred_file, sep="\t", index=False)


if __name__ == "__main__":
    main(cnn_models.CNN_COMB())
