import torch 
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from timeit import default_timer as timer
import os
import numpy as np
from PIL import Image

images = []
path = os.curdir + '/images'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# for image in os.listdir(path):
#     try:
#         input_image = Image.open(path+'/'+image)
#         images.append(preprocess(input_image).unsqueeze(0))
#     except:
#         continue

alexnet_model = models.alexnet(pretrained=True)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

all_layers = list(alexnet_model.features.children()) + \
    [nn.AdaptiveAvgPool2d((6, 6)), Flatten()] + \
    list(alexnet_model.classifier.children())

class SingleLayer(nn.Module):
    def __init__(self, layer_index):
        super(SingleLayer, self).__init__()
        self.features = nn.Sequential(all_layers[layer_index])

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":

    # prev_output = images

    layers = [SingleLayer(i) for i in range(0, 22)]
    time = [0. for i in range(0,22)]

    for j, image in enumerate(os.listdir(path)):
        curr_image = preprocess(Image.open(path+'/'+image)).unsqueeze(0)
        time_taken = 0.0
        for i, layer in enumerate(layers):
            start = timer()

            curr_image = layer(curr_image)

            end = timer()

            time[i] += (end-start)
            time_taken += (end-start)

        print(j, time_taken)

    np.divide(time, 9812)

    for i, t in enumerate(time):
        print(layers[i])
        print('Layer', i, t)