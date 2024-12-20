import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    """
    This is specific for MNIST.
    """

    def __init__(self, tau: float = 1):
        super(Net, self).__init__()
        self.tau = tau

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        # that's pretty specific to MNIST...
        self.fc = nn.Sequential(
            nn.Linear(in_features=320, out_features=50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=50, out_features=10),

        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 320)
        x = self.fc(x)
        return F.log_softmax(x, dim = 1), F.softmax(x / self.tau, dim = 1), x
    
    
class MLP(nn.Module):
    def __init__(self, tau: float = 1.0):
        super(MLP, self).__init__()
        self.tau = tau
        # Define the model with two hidden layers (1,024 neurons each)
        self.fc1 = nn.Linear(28*28, 1024)  # First hidden layer (28x28 pixels for Fashion-MNIST)
        self.relu1 = nn.ReLU()             # ReLU activation function
        self.fc2 = nn.Linear(1024, 1024)   # Second hidden layer
        self.relu2 = nn.ReLU()             # ReLU activation function
        self.fc3 = nn.Linear(1024, 10)     # Output layer (10 classes for Fashion-MNIST)

    def forward(self, x):
        # Flatten the image (28x28 pixels) into a vector of 784 values
        x = x.view(-1, 28*28)
        x = self.fc1(x)        # Pass through the first layer
        x = self.relu1(x)      # ReLU activation
        x = self.fc2(x)        # Pass through the second layer
        x = self.relu2(x)      # ReLU activation
        x = self.fc3(x)        # Pass through the output layer
        
        # Output three versions as in the `Net` class:
        # - log softmax for the standard classification loss (for training)
        # - softmax with temperature scaling (for regularization or knowledge distillation)
        # - raw logits (useful for later layers or loss functions)
        return F.log_softmax(x, dim=1), F.softmax(x / self.tau, dim=1), x
    
    
class ResNet18(nn.Module):
   
    def __init__(self, num_classes=10, tau=1.):
        super(ResNet18, self).__init__()
        self.tau = tau
        # Load pre-trained ResNet-18
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Modify the final fully connected layer to output 128 features
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 128)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.float()
        # Pass through the ResNet base model
        x = self.base_model(x)
        # Pass through the custom classifier
        x = self.classifier(x)
        return F.log_softmax(x, dim=1), F.softmax(x / self.tau, dim=1), x
    

class ResNet50(nn.Module):
   
    def __init__(self, num_classes=10, tau=1.):
        super(ResNet50, self).__init__()
        self.tau = tau
        # Load pre-trained ResNet-18
        self.base_model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Modify the final fully connected layer to output 128 features
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 128)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.float()
        # Pass through the ResNet base model
        x = self.base_model(x)
        # Pass through the custom classifier
        x = self.classifier(x)
        return F.log_softmax(x, dim=1), F.softmax(x / self.tau, dim=1), x
    

class ResNet18Embedder(nn.Module):

    def __init__(self):
        super(ResNet18Embedder, self).__init__()
        resnet50 = models.resnet50(weights='IMAGENET1K_V1')        
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        # freezing the parameters of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        return self.backbone(x.float()).view(batch_size, -1)


def get_network(dataset: str) -> nn.Module:
    """
    """
    if dataset in ['mnist']:
        return Net()
    if dataset in ['fashion-mnist']:
        return MLP()
    if dataset in ['cifar10']:
        return ResNet50()
    raise Exception("dataset not recognised.")

def get_embedder(dataset: str) -> nn.Module:
    if dataset in ['mnist', 'fashion-mnist']:
        return None
    if dataset in ['cifar10']:
        return ResNet18Embedder()
    raise Exception("dataset not recognised.")
