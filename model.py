import torch
import torch.nn as nn
import sys

path = '../pytorch_model_weights/'
sys.path.append(path)

from torchvision.models import resnet18, resnet50, efficientnet_b0, efficientnet_b3

class ResNet18(nn.Module):
    def __init__(self, nclasses):
        super(ResNet18, self).__init__()
        
        # Load the pre-trained ResNet18 model
        self.model = resnet18(weights=None)
        
        weights = torch.load(path + 'resnet18-IMAGENET1K_V1.pth')
        self.model.load_state_dict(weights)
        
        for param in self.model.parameters():
            param.requires_grad = False
                
        # Modify the output layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, nclasses)
        
    
    def forward(self, x):
        return self.model(x)



class ResNet50(nn.Module):
    def __init__(self, nclasses):
        super(ResNet50, self).__init__()
        
        # Load the pre-trained ResNet18 model
        self.model = resnet50(weights=None)
        
        weights = torch.load(path + 'resnet50-IMAGENET1K_V1.pth')
        self.model.load_state_dict(weights)
        
        for param in self.model.parameters():
            param.requires_grad = False
                
        # Modify the output layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, nclasses)
    
    def forward(self, x):
        return self.model(x)
    
    

class EfficientNet_b0(nn.Module):
    def __init__(self, nclasses):
        super(EfficientNet_b0, self).__init__()
        
        self.model = efficientnet_b0(weights=None)
        
        weights = torch.load(path + 'efficientnet_b0_IMAGENET1K_V1.pth')
        self.model.load_state_dict(weights)
        
        for param in self.model.parameters():
            param.requires_grad = False
        

        # Modify the output layer for n classification
        linear_layer = self.model.classifier[1]
        num_ftrs = linear_layer.in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, nclasses)
       
        
    def forward(self, x):
        return self.model(x)



class EfficientNet_b3(nn.Module):
    def __init__(self, nclasses):
        super(EfficientNet_b3, self).__init__()
        
        self.model = efficientnet_b3(weights=None)
        
        weights = torch.load(path + 'efficientnet_b3_IMAGENET1K_V1.pth')
        self.model.load_state_dict(weights)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the output layer for n classification
        linear_layer = self.model.classifier[1]
        num_ftrs = linear_layer.in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, nclasses)
        
    def forward(self, x):
        return self.model(x)
    
    
    