import torch, tqdm
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from word_embedding import CODE2TENSOR
from variables import *

def conv_block(in_channels, out_channels, final_layer = False):
    if not final_layer:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=2
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                        )
                    )  
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=2
                ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                        )
                    )    

def linear_block():
    return nn.Sequential(
                nn.Flatten(),
                nn.Linear(7680, 512)
                        )          

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64, 128)
        self.conv5 = conv_block(128, 64, True)
        self.linear = linear_block()

        self.cnn = nn.Sequential(
                            self.conv1,
                            self.conv2,
                            self.conv3,
                            self.conv4,
                            self.conv5,
                            self.linear
                                )   

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, lecturer_embedding, student_embedding):
        fv1 = self.cnn(lecturer_embedding)
        fv2 = self.cnn(student_embedding)

        fv1 = fv1.view(1, -1)
        fv2 = fv2.view(1, -1)
        sim_score = self.cos_sim(fv1, fv1).squeeze()

        return sim_score

class AutomaticGrading(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = CODE2TENSOR()

        self.X, self.Y, self.Errors = data.code_embedding()  

        model = SiameseNetwork()
        self.model = model.to(self.device)
        self.data = data

        input_shape = torch.unsqueeze(self.data.__getitem__(0)[0][0,:,:], 0).shape
        summary(self.model, [input_shape, input_shape])


        # self.optimizer = optim.Adam(
        #                         self.cnn.parameters(), 
        #                         lr=learning_rate
        #                             )
        # self.loss_fn = nn.CrossEntropyLoss()

model = AutomaticGrading()