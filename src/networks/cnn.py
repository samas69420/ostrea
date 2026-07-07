import torch

class CNN(torch.nn.Module):

    def __init__(self, in_channels, device = torch.device("cpu")):

        super().__init__()
        
        self.net = torch.nn.Sequential(
                       torch.nn.Conv2d(in_channels, 64, kernel_size = 8, stride = 4, padding = "valid"),
                       torch.nn.ReLU(),
                       torch.nn.Conv2d(64, 128, kernel_size = 4, stride = 3, padding = "valid"),
                       torch.nn.ReLU(),
                       torch.nn.Conv2d(128, 128, kernel_size = 4, stride = 2, padding = "valid"),
                       torch.nn.ReLU(),
                       torch.nn.Flatten()).to(device)

    def forward(self, x):
        
        return self.net(x)
