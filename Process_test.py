#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import numpy as np
import matplotlib as plt
import os
import math
import torchvision
from torchvision import datasets, transforms, models
import torch.optim as optim
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset, DataLoader
import copy
import math
from dataLoad import ED_dataset
from PIL import Image

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.conv1.in_planes=in_channels
        self.conv1.out_planes=out_channels
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class ConvDown(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvDown, self).__init__()

        self.conv1=SeparableConv2d(in_planes, out_planes, 3, stride=2, padding=1)
        self.in_planes= in_planes
        self.out_planes= out_planes
        self.bn1= nn.BatchNorm2d(out_planes)
        self.bn1.out_planes=out_planes
        self.relu1= nn.ReLU(inplace=True)
        
    def forward(self, x):
        out= self.relu1(self.bn1(self.conv1(x)))
        return out
        
class UpConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(UpConv, self).__init__()
        self.conv1= nn.ConvTranspose2d(in_planes, out_planes, 4, stride=2, padding=1)
        self.in_planes= in_planes
        self.out_planes= out_planes
        self.bn1= nn.BatchNorm2d(out_planes)
        self.relu1= self.relu1= nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        out= self.relu1(self.bn1(self.conv1(x)+skip))
        return out
    
class FlatConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FlatConv, self).__init__()
        self.conv1= SeparableConv2d(in_planes, out_planes, 3, stride=1, padding=1)
        self.in_planes= in_planes
        self.out_planes= out_planes
        self.bn1= nn.BatchNorm2d(out_planes)
        self.relu1= self.relu1= nn.ReLU(inplace=True)
        
    def forward(self, x, last=None, skipin=None):
        if last!=None:
            skip= self.bn1(self.conv1(x))
            out= self.relu1(skip)
            return out, skip
        else:
            if skipin!= None:
                out= self.relu1(self.bn1(self.conv1(x)+skipin))
            else:
                out= self.relu1(self.bn1(self.conv1(x)))
            return out
        
class RGB_model(nn.Module):
    def __init__(self):
        super(RGB_model, self).__init__()
        self.F0 = SeparableConv2d(5, 21, 5, padding=2) 
        self.D1= ConvDown(21, 21)
        self.F1_1= FlatConv(21, 42)
        self.F1_2= FlatConv(42, 42)
        self.D2= ConvDown(42, 84)
        self.F2_1= FlatConv(84, 84)
        self.F2_2= FlatConv(84, 84)
        self.F2_3= FlatConv(84, 84)
        self.D3= ConvDown(84, 168)
        self.F3_1= FlatConv(168, 168)
        self.F3_2= FlatConv(168, 168)
        self.F3_3= FlatConv(168, 168)
        self.U1= UpConv(168, 84)
        self.F4_1= FlatConv(84, 84)
        self.F4_2= FlatConv(84, 84)
        self.F4_3= FlatConv(84, 84)
        self.U2= UpConv(84, 42)
        self.F5_1= FlatConv(42, 42)
        self.F5_2= FlatConv(42, 21)
        self.U3= UpConv(21, 21)
        self.F6_1= FlatConv(21, 5)
        self.F6_2 = SeparableConv2d(5, 1)
        
        self.batch5 = nn.BatchNorm2d(5)
        self.batch21 = nn.BatchNorm2d(21)

    def custom(self, module):
        def custom_forward(*inputs,  **kwargs):
            if len(kwargs)==1:
                inputs= module(inputs[0], kwargs[0])
            elif len(inputs)==2:
                inputs= module(inputs[0], inputs[1])
            else:
                inputs = module(inputs[0])
            return inputs
        return custom_forward
    
    def forward(self, inp):
        skip1= inp
        skip2= self.batch21(self.F0(inp))
        x= F.relu(skip2)
        x= checkpoint.checkpoint(self.custom(self.D1), x) 
        x= self.F1_1(x)
        x, skip3= self.F1_2(x, last=skip1)
        x= checkpoint.checkpoint(self.custom(self.D2), x)
        x= self.F2_1(x)
        x= self.F2_2(x)
        x, skip4= self.F2_3(x, last=skip1)
        x= checkpoint.checkpoint(self.custom(self.D3), x)
        x= self.F3_1(x)
        x= self.F3_2(x)
        x= self.F3_3(x)
        x= checkpoint.checkpoint(self.custom(self.U1), x, skip4)
        x= self.F4_1(x)
        x= self.F4_2(x)
        x= self.F4_3(x)
        x= checkpoint.checkpoint(self.custom(self.U2), x, skip3)
        x= self.F5_1(x)
        x= self.F5_2(x)
        x= checkpoint.checkpoint(self.custom(self.U3), x, skip2)
        x= self.F6_1(x, skipin= skip1)
        x= self.F6_2(x)
      
        return x
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.R= RGB_model()
        self.G= RGB_model()
        self.B= RGB_model()
        
    def forward(self, inp):
        
        R= inp[:,(0,3,6,9,12),:,:]
        G= inp[:,(1,4,7,10,13),:,:]
        B= inp[:,(2,5,8,11,14),:,:]
        
        Rout= self.R(R)
        Gout= self.G(G)
        Bout= self.B(B)
        
        out= torch.cat((Rout,Gout,Bout),1)
        
        return out
    

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('UpConv') !=-1:
        m.conv1.weight.data.normal_(mean=0.0, std= math.sqrt(2./max(m.in_planes, m.out_planes)))
        if m.conv1.bias is not None:
            m.conv1.bias.data.fill_(0)
        m.bn1.weight.data.normal_(mean=0.0, std= math.sqrt(2./m.out_planes))
        
    if classname.find('ConvDown') !=-1:
        m.conv1.conv1.weight.data.normal_(mean=0.0, std= math.sqrt(2./max(m.in_planes, m.out_planes)))
        if m.conv1.conv1.bias is not None:
            m.conv1.conv1.bias.data.fill_(0)
        m.bn1.weight.data.normal_(mean=0.0, std= math.sqrt(2./m.out_planes))
    if classname.find('FlatConv') !=-1:
        m.conv1.conv1.weight.data.normal_(mean=0.0, std= math.sqrt(2./max(m.in_planes, m.out_planes)))
        if m.conv1.conv1.bias is not None:
            m.conv1.conv1.bias.data.fill_(0)
        m.bn1.weight.data.normal_(mean=0.0, std= math.sqrt(2./m.out_planes))
    if classname.find('bn1') !=-1:
        m.weight.data.normal_(mean=0.0, std= math.sqrt(2./m.out_planes))
        
        
def imsave(image, folder, num):
    Rmean= 111.35506916879196
    Gmean= 110.87522703527382
    Bmean= 107.35623607453486
    Rstd= 62.48659498278817
    Gstd= 61.90064882948136
    Bstd= 66.58884127671352
    
    image[0]= image[0]*Rstd+Rmean
    image[1]= image[1]*Gstd+Gmean
    image[2]= image[2]*Bstd+Bmean
    image= np.moveaxis(image, 0, -1)
    im= Image.fromarray(np.uint8(image))
    filepath= "../../../../usr/project/xtmp/dys9/test/test_sharp"
    
        
    if num>9:
        num= '000000'+str(num)
    else:   
        num= '0000000'+str(num)
        
    path= filepath+ str(folder)+'/'+num+'.png'
    
    im.save(path)
    
    return None
        
    
    
    
def test_net():

    net= Net()
    net.load_state_dict(torch.load('ED_RGB_final.pt', map_location=torch.device('cpu')))
    net.eval()

    net.to(cuda)
    
    filepath= "../../../../usr/project/xtmp/dys9"
    
    total_time=0
    
    for i in range(1):
        
        folder= i//100
        image= (i%100)
        
        blurFull= np.zeros((3, 720, 1280))
        
        for h in range(5):
            for w in range(5):
                blpath= filepath+'/'+'test'+'/'+'test'+'_blur'+str(folder)+'/'+str(image)+'_crop'+str(h*5+w)+'.npy'
                blur= np.load(blpath)
                
                blurred= blur
                blurred= np.expand_dims(blurred, axis=0)
                blurred= torch.tensor(blurred)
                blurred= blurred.float()
                blurred=blurred.cuda()
                
                t1= time.time()
                out=net(blurred)
                t2=time.time()
                
                out= out.cpu().detach().numpy()
                blurFull[:,h*144:h*144+144,w*256:w*256+256]= out
                
                total_time+= t2-t1
                
        imsave(blurFull, folder, image)
                
    avg_time= total_time
 
    return total_time

if __name__ == '__main__':
    cuda = torch.device('cuda')
    start_time = time.time()
    avg_time= test_net()
    print('average time per image= '+str(avg_time))
    print('total time= '+ str(time.time-start_time))


