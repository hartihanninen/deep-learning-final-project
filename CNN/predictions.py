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

def main(model: nn.Module):
    """Start the CNN model."""
    annotations = ["baby",
                "bird",
                "car",
                "clouds",
                "dog",
                "female",
                "flower",
                "male",
                "night",
                "people",
                "portrait",
                "river",
                "sea",
                "tree"]
    cnn.create_labels()
    if NEW_SPLIT:
        cnn.split_data()
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
    loss_function = nn.BCEWithLogitsLoss()

    cnn.test(model, loss_function, device, pred_loader, annotations)


def transform_data(mean: float, std: float):
    transform = transforms.Compose([            
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize(256),                    
                                        transforms.CenterCrop(224),                
                                        transforms.ToTensor(),                     
                                        transforms.Normalize(                      
                                        mean=mean,                
                                        std=std
                                        )])
    pred_set = cnn.myDataset(PRED_DIR, transform=transform)
    pred_loader = torch.utils.data.DataLoader(dataset=pred_set, batch_size=1, shuffle=False)
    return pred_loader