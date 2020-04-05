import torch 
import torch.nn as nn
import torchvision
from torchvision import models
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('partition_layer', action="store", type=int)
parser.add_argument('-cloud', action='store_true', default=False,
                    dest='cloud')

images = torch.zeros(9,3,224,224)

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

class MobileAlexNet(nn.Module):
    def __init__(self, partition_layer):
        super(MobileAlexNet, self).__init__()
        self.features = self.get_layers(partition_layer)

    def get_layers(self, partition_layer):
        return nn.Sequential(*list(all_layers[:partition_layer]))
    
    def forward(self, x):
        x = self.features(x)
        return x

class ServerAlexNet(nn.Module):
    def __init__(self, partition_layer):
        super(ServerAlexNet, self).__init__()
        self.features = self.get_layers(partition_layer)

    def get_layers(self, partition_layer):
        return nn.Sequential(*list(all_layers[partition_layer:]))
    
    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":
    args = parser.parse_args()
    partition_layer, is_cloud = args.partition_layer, args.cloud

    if not is_cloud:
        #mobile device
        net = MobileAlexNet(partition_layer)
        prev_output = images
        print('\nMobile with following Layers :')
        for i, layer in enumerate(all_layers[:partition_layer]):
            print(i , layer)
    else:
        #cloud
        net = ServerAlexNet(partition_layer)
        mobile_net = MobileAlexNet(partition_layer)
        prev_output = mobile_net(images)
        print('\nCloud with following Layers :')
        for i, layer in enumerate(all_layers[partition_layer:]):
            print(partition_layer + i , layer)
    
    t = 0.0
    for i in range(5):
        start = timer()

        output = net(prev_output)

        end = timer()

        print('Run ', i, ':', end - start)

        t += (end - start)
    
    print('\nAverage Time Taken:', t/5)