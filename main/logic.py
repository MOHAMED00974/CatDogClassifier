from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features= nn.Sequential(

            nn.Conv2d(3, 32, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2),

            nn.Conv2d(32, 64, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2),

            nn.Conv2d(64, 128, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2),
        )

        self.classifier= nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x= self.features(x)
        x= self.classifier(x)
        return x

model= CNN()
state_dict = torch.load('MakingTheModel/model.pt', weights_only=True, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

def predict(image: Image)-> str: 
    '''
        This method is doing 2 things:
        1- it preprocess the input
        2- it passes it to the model, then returns the label
    '''
    image= image.convert('RGB')
    transform= (Compose([Resize((224, 224)), ToTensor()]))
    image= transform(image).unsqueeze(0)
    labels= model(image)
    if labels.argmax()== 1: 
        return "Dog"
    else: 
        return "Cat"
