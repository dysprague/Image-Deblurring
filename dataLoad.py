from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch

class ED_dataset(Dataset):
    
    def __init__(self, filepath, which):
        
        self.filepath= filepath
        self.which= which
        
    def __len__(self):
        
        if (self.which=='train'):
            return 120*16*25
        if (self.which=='val'):
            return 30*100*25
        
        return 0
        
    def __getitem__(self, idx):
        
        folder= idx//400+120
        image= (idx%400)//25
        crop= (idx%25)
        
        blpath= self.filepath+'/'+self.which+'/'+self.which+'_blur_proc/'+'s'+str(folder)+'/'+str(image)+'_crop'+str(crop)+'.npy'
        shpath= self.filepath+'/'+self.which+'/'+self.which+'_shar_proc/'+'s'+str(folder)+'/'+str(image)+'_crop'+str(crop)+'.npy'
        
        blur= np.load(blpath)
        shar= np.load(shpath)
            
        return blur, shar
        
class RNN_dataset(Dataset):
    
    def __init__(self, filepath, which):
        
        self.filepath= filepath
        self.which= which
        
    def __len__(self):
        
        if (self.which=='train'):
            return 240*25
        if (self.which=='val'):
            return 30*25
        
        return 0
    
    def __getitem__(self, idx):
        
        folder= idx//25
        crop= idx%25
        
        blur= np.zeros((100, 3, 144, 256))
        shar= np.zeros((100, 3, 144, 256))
        
        for i in range(100):
            blpath= self.filepath+'/'+self.which+'/'+self.which+'_blur_proc/'+'s'+str(folder)+'/'+str(image)+'_crop'+str(crop)+'.npy'
            shpath= self.filepath+'/'+self.which+'/'+self.which+'_shar_proc/'+'s'+str(folder)+'/'+str(image)+'_crop'+str(crop)+'.npy'
            
            blur[i, :, :, :]= np.load(blpath)[6:9,:,:]
            shar[i, :, :, :]= np.load(shpath)
            
        return blur, shar
        
