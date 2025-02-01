import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTAdditionCNN(nn.Module):
    """
        Processes to images stacked together as pair, returns the sum as 19 class output
        Input (1, 28, 2*28)--> Conv (16 filters) --> ReLU --> Max Pooling (2x2) --> (16, 14, 28)
              (16, 14, 28) --> Conv (32 filters) --> ReLU --> Max Pooling (2x2) --> (32, 7, 14)
              (32, 7, 14)  --> Flattened to (3136)

              (3136) --> FC1 (128 units) --> FC2 (64 units) --> FC3 (19 units)
              (19 units) --> Softmax (output probabilities)
        """
    def __init__(self):
        super(MNISTAdditionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 7 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


if __name__ == "__main__":
    model = MNISTAdditionCNN()
    dummy_input = torch.randn(64, 1, 28, 56)
    output = model(dummy_input)
    print(output.shape)
