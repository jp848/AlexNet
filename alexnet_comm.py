import torch 
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from timeit import default_timer as timer
import os
import numpy as np
from PIL import Image
import argparse
import socket
import pickle
from functools import reduce
import os

parser = argparse.ArgumentParser()
parser.add_argument('-cloud', action='store_true', default=False,
                    dest='cloud')
# parser.add_argument('layer')

images = []
path = os.curdir + '/images'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

    args = parser.parse_args()
    is_cloud = args.cloud
    layers = [SingleLayer(i) for i in range(0, 22)]

    if not is_cloud:
        TCP_IP1 = '10.0.2.2'
        TCP_PORT = 8080
        
        process_time = 0.0
        comm_time = 0.0
        for layer in range(0,22):
            for i, image in enumerate(os.listdir(path)):
                input_image = preprocess(Image.open(path+'/'+image)).unsqueeze(0)

                process_start = timer()

                output = reduce(lambda o, func: func(o), layers[:layer], input_image)

                process_end = timer()

                #send to server
                comm_start = timer()

                s=socket.socket()
                s.connect((TCP_IP1,TCP_PORT))
                send_x=output.detach().numpy()
                data_input=pickle.dumps(send_x, protocol=pickle.HIGHEST_PROTOCOL)
                s.sendall(data_input)
                s.close()
                
                comm_end = timer()

                process_time += (process_end - process_start)
                comm_time += (comm_end - comm_start)

                # print('Image #' + str(i) + ' sent.')

            print(layer, "Avg. Process Time:", process_time/len(os.listdir(path)))
            print(layer, "Avg. Comm Time:", comm_time/len(os.listdir(path)))

    else:
        TCP_IP2 = '127.0.0.1'
        TCP_PORT = 8080
        s=socket.socket()
        
        s.bind((TCP_IP2, TCP_PORT))
        cnt, process_time = 0, 0.0
        layer = 0
        
        while 1:
            cnt += 1
            s.listen(1)

            conn, addr = s.accept()
            data = []
            while 1:
                tensor = conn.recv(4096)
                if not tensor: break
                data.append(tensor)
            inputs_ten = pickle.loads(b"".join(data))

            output_mobile = torch.from_numpy(inputs_ten)

            print("Image #" + str(cnt) + ' received.')

            process_start = timer()

            output = reduce(lambda o, func: func(o), layers[layer:], output_mobile)

            process_end = timer()

            process_time += (process_end - process_start)

            if cnt % len(os.listdir(path)) == 0:
                print("Avg. Process Time:", process_time/cnt)
                os.system('say "your program has finished"')
                layer += 1