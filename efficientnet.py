import torch
import torch.nn as nn
import sys
from summary import Summary

path = '../pretrained_models/EfficientNet/'
sys.path.append(path)

from torchvision.models import efficientnet_b0


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
    

def main():
    # Creating model and testing output shapes 
    model = EfficientNet_b0(nclasses=100) 
    Summary(model)
    
    

if __name__ == "__main__": 
    main()
    
    