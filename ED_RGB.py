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
   
    
def train_net(num_epochs, batch_size, mbs):

    net= Net()
    net.load_state_dict(torch.load('ED_RGB_full120_16_10_64_16.pt', map_location=torch.device('cpu')))
    
    print('epochs= '+str(num_epochs))
    print('batch size= '+str(batch_size))
    print('mbs= '+str(mbs))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(cuda)
    
    filepath= "../../../../usr/project/xtmp/dys9"
    
    train_data= ED_dataset(filepath, 'train')
    
    lr = .005
    
    optimizer= optim.Adam(net.parameters(), lr, betas= (0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [6,10,14,18,22,26], gamma=0.25)
    
    losses= np.zeros(num_epochs)
    train_loader= DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1) #change num_workers here
    for i in range(num_epochs):
        t1= time.time()
        epoch_loss, net = iterate_through_batches(train_loader, mbs, batch_size, net, optimizer) # pass optim and net as args
        scheduler.step()
        t2= time.time()
        print('Single Epoch time'+str(t2-t1))
        print(epoch_loss)
        losses[i]= epoch_loss
      
    name = 'ED_RGB_retrain'+'_'+str(num_epochs) + '_' + str(batch_size) + '_'+str(mbs)+'.pt' 
    torch.save(net.state_dict(), name)
    
    return losses
    
def iterate_through_batches(dataloader, mbs, batch_size, net, optimizer):
    
    epoch_loss= 0
    for blur, sharp in dataloader:
        
        #netTrain= copy.deepcopy(net)
        #netTrain.half()
        #for layer in netTrain.modules():
        #    for mod in layer.modules():
        #        if isinstance(mod, nn.BatchNorm2d):
        #            mod.float()
        
        #if((i < 116 and i > 30) and (i % 12 == 8)) or ((i > 106 ) and (i % 18 == 0)):
        #    lr = lr/2
        #    optimizer = optim.Adam(net.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        
        blur= blur.cuda().float()#.half()
        sharp= sharp.cuda().float()#.half()
        
        for v in range(0, batch_size//mbs):

            out = net(blur[v*mbs:v*mbs+mbs,:,:,:])
            target = sharp[v*mbs:v*mbs+mbs,:,:,:]
            criterion = nn.MSELoss()  

            loss = criterion(out, target)
            
            loss.backward()
            
        loss.float()
        optimizer.step()  
        optimizer.zero_grad()  
        epoch_loss += loss.item()
        torch.cuda.empty_cache()
          
    return epoch_loss, net

if __name__ == '__main__':
    cuda = torch.device('cuda')
    start_time = time.time()
    losses= train_net(10, 64, 16)
    print(losses)
    print("%s" % (time.time()-start_time))

