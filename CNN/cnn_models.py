import torch.nn as nn
from torchvision import transforms, datasets, models
import torch


class CNN_COMB(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(CNN_COMB, self).__init__()
        
        self.alex = models.alexnet(pretrained=True)
        for p in self.alex.parameters():
            p.requires_grad=False
        
        self.resnet = models.resnet18(pretrained=True)
        for p in self.resnet.parameters():
            p.requires_grad=False
        
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 20, (5,5)),
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
            nn.Linear(250, num_classes)
        )

    def forward(self, x):
        
        out_alex = self.alex(x)
        out_resnet = self.resnet(x)
        out_conv = self.conv(x)
        out_conv = self.flatten(out_conv)
        out_conv = self.lin_conv(out_conv)
        
        combined_output = torch.cat((out_alex, out_resnet, out_conv), dim=1)
        
        out = self.lin(combined_output)
        
        return out