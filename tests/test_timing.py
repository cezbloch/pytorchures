
from object_detection.timing import wrap_model_layers, TimedLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # self.conv1 = TimedLayer(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1))
        # self.conv2 = TimedLayer(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
        # self.fc1 = TimedLayer(nn.Linear(32 * 7 * 7, 128))
        # self.fc2 = TimedLayer(nn.Linear(128, 10))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # Apply second convolutional layer followed by ReLU and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the tensor
        x = x.view(-1, 32 * 7 * 7)
        # Apply the fully connected layer followed by ReLU
        x = F.relu(self.fc1(x))
        # Apply the output layer
        x = self.fc2(x)
        return x



def test_layer_timing():
    # model = torch.nn.Sequential(torch.nn.Linear(2, 2), 
    #                       torch.nn.ReLU(),
    #                       torch.nn.Sequential(torch.nn.Linear(2, 1),
    #                       torch.nn.Sigmoid()))

    # Instantiate the model
    model = SimpleCNN()

    # Print the model architecture
    print(model)

    # Create a random tensor with the shape of (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)  # Example: batch of 1 grayscale image of 28x28 pixels

    wrap_model_layers(model)
    
    # Run the model with the input tensor
    output = model(input_tensor)
    print(output)