import torch.nn as nn
from torchvision import transforms, datasets, models
import torch
from constants import NUM_CLASSES, NUM_CHANNELS


class CNN_COMB(nn.Module):
    """CNN combining pretrained AlexNet, ResNet, and a custom model."""

    def __init__(self, num_classes=NUM_CLASSES):
        """Initialize AlexNet and ResNet with pre-trained values and do not update these."""
        super(CNN_COMB, self).__init__()

        self.alex = models.alexnet(pretrained=True)
        for p in self.alex.parameters():
            p.requires_grad=False

        self.resnet = models.resnet18(pretrained=True)
        for p in self.resnet.parameters():
            p.requires_grad=False

        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 20, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride = (2,2)),

            nn.Conv2d(20, 50, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride = (2,2)),

            nn.Conv2d(50, 100, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride = (2,2))
        )
        self.flatten = nn.Flatten()
        self.lin_conv = nn.Sequential(
            nn.Linear(57600, 1000),
            nn.ReLU()
        )

        self.lin = nn.Sequential(
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, NUM_CLASSES)
        )

    def forward(self, x):
        """Do the forward propagation."""
        out_alex = self.alex(x)
        out_resnet = self.resnet(x)
        out_conv = self.conv(x)
        out_conv = self.flatten(out_conv)
        out_conv = self.lin_conv(out_conv)

        combined_output = torch.cat((out_alex, out_resnet, out_conv), dim=1)

        out = self.lin(combined_output)

        return out


class CNN_FIRST(nn.Module):
    """Simple CNN."""

    def __init__(self, num_classes=NUM_CLASSES):
        """Define the convolutional and linear parts."""
        super(CNN_FIRST, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 20, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride = (2,2)),

            nn.Conv2d(20, 50, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride = (2,2)),

            nn.Conv2d(50, 100, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride = (2,2))
        )
        self.flatten = nn.Flatten()
        self.lin = nn.Sequential(
            nn.Linear(14400, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        """Do the forward propagation."""
        x = self.conv(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x



class CNN_BASE(nn.Module):
    """CNN with multiple convolutional layers before maxpooling."""

    def __init__(self, num_classes=NUM_CLASSES):
        """Define the convolutional and linear parts."""
        super(CNN_BASE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.flatten = nn.Flatten()
        self.lin = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(65536, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Do the forward propagation."""
        x = self.conv(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x