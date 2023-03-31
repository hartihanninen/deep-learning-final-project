import torch.nn as nn
from torchvision import transforms, datasets, models
import torch
from constants import NUM_CLASSES, NUM_CHANNELS


class CNN_COMB(nn.Module):
    """CNN combining pretrained AlexNet, ResNet, and a custom model."""

    def __init__(self):
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
