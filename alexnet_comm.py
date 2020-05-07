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

parser = argparse.ArgumentParser()
parser.add_argument('-cloud', action='store_true', default=False,
                    dest='cloud')
parser.add_argument('-layer', type=int)

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

    # prev_output = images

    # layers = [SingleLayer(i) for i in range(0, 22)]
    # time = [0. for i in range(0,22)]

    # for j, image in enumerate(os.listdir(path)):
    #     curr_image = preprocess(Image.open(path+'/'+image)).unsqueeze(0)
    #     time_taken = 0.0
    #     for i, layer in enumerate(layers):
    #         start = timer()

    #         curr_image = layer(curr_image)

    #         end = timer()

    #         time[i] += (end-start)
    #         time_taken += (end-start)

    #     print(j, time_taken)

    # np.divide(time, 9812)

    # for i, t in enumerate(time):
    #     print(layers[i])
    #     print('Layer', i, t)

    args = parser.parse_args()
    is_cloud, layer = args.cloud, args.layer

    TCP_IP1 = '10.0.2.2'
    TCP_IP2 = '127.0.0.1'
    TCP_PORT = 8080
    s=socket.socket()

    if not is_cloud:
        outputs1 = torch.zeros(9,3,224,224)
        s.connect((TCP_IP1,TCP_PORT))
        send_x=outputs1.detach().numpy()
        data_input=pickle.dumps(send_x, protocol=pickle.HIGHEST_PROTOCOL)
        s.sendall(data_input)
        s.close()

        print("data sent to server")

        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP2, TCP_PORT))
        print('trying to listen')
        s.listen(1)

        conn, addr = s.accept()
        data = []
        print('Server: ', addr)
        while 1:
            output = conn.recv(4096)
            if not output: break
            data.append(output)
        
        print('received from server')

    else:
        s.bind((TCP_IP2, TCP_PORT))
        cnt = 0
        while 1:
            cnt += 1
            s.listen(1)
            print('server started', cnt)

            conn, addr = s.accept()

            print ('Raspberry Device:',addr)
            
            data = []
            while 1:
                tensor = conn.recv(4096)
                if not tensor: break
                data.append(tensor)
            inputs_ten = pickle.loads(b"".join(data))

            conn.close()
            output = torch.from_numpy(inputs_ten)

            print("data received by server")

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP1, TCP_PORT))
            print('trying to send')
            data_final=pickle.dumps(output,protocol=pickle.HIGHEST_PROTOCOL)
            s.sendall(data_final)
            print('server done')