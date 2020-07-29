from PIL import Image
import numpy as np
import os
from IPython.display import display
import math

def getFile(folder, image):

    cwd= os.getcwd()
    filepath= "../../../../usr/xtmp/bbp13"
    blur= filepath+ '/test/test_blur'
    
    if folder>99:
        folder= str(folder)
    elif folder>9:
        folder= '0'+str(folder)
    else:
        folder= '00'+str(folder)
        
    if image>9:
        image= '000000'+str(image)
    else:   
        image= '0000000'+str(image)

    fileblur= blur+'/'+folder+'/'+image+'.png'

    imblur=Image.open(fileblur)
    
    np_imblur= np.array(imblur, dtype='float')
    
    return np_imblur

def get_mean_std(folders):
    
    Rmean= 0
    Gmean= 0
    Bmean= 0
    Rvar= 0
    Gvar= 0
    Bvar= 0
    
    for i in range(folders):
        for j in range(100):
            imbl, imsh= getFile(i, j)
            
            imbl= np.moveaxis(imbl, -1, 0)
            imsh= np.moveaxis(imsh, -1, 0)
            
            Rmean+=np.mean(imbl[0,:,:])
            Gmean+=np.mean(imbl[1,:,:])
            Bmean+=np.mean(imbl[2,:,:])
            
            Rvar+=np.std(imbl[0,:,:])**2
            Gvar+=np.std(imbl[1,:,:])**2
            Bvar+=np.std(imbl[2,:,:])**2
          
    Rmean= Rmean/(folders*100)
    Gmean= Gmean/(folders*100)
    Bmean= Bmean/(folders*100)
    Rstd= math.sqrt(Rvar/(folders*100))
    Gstd= math.sqrt(Gvar/(folders*100))
    Bstd= math.sqrt(Bvar/(folders*100))
    
    return Rmean, Gmean, Bmean, Rstd, Gstd, Bstd

def normalize_save(folders, which):
    
    #Rm, Gm, Bm, Rs, Gs, Bs=get_mean_std(folders)
    Rm= 111.35506916879196
    Gm= 110.87522703527382
    Bm= 107.35623607453486
    Rs= 62.48659498278817
    Gs= 61.90064882948136
    Bs= 66.58884127671352
    
    filepath= "../../../../usr/project/xtmp/dys9"
    
    for i in range(folders):
        for j in range(100):
            
            blur= np.zeros((15, 720, 1280))
           
            for k in range(5):
                curr= j-2+k

                if curr==-2:
                    curr=0
                if curr==-1:
                    curr=0
                if curr==100:
                    curr= 99
                if curr== 101:
                    curr= 99

                imbl= getFile(i, curr)
                
                imbl= np.moveaxis(imbl, -1, 0)
                
                imbl[0,:,:]= (imbl[0,:,:]-Rm)/Rs
                imbl[1,:,:]= (imbl[1,:,:]-Gm)/Gs
                imbl[2,:,:]= (imbl[2,:,:]-Bm)/Bs
                
                blur[k*3:k*3+3,:,:]= imbl
               
            
            for h in range(5):
                for w in range(5):
                    blcrop= blur[:,h*144:h*144+144,w*256:w*256+256]
            
                    blFile= filepath+'/'+which+'/'+which+'_blur'+str(i)
                    blName= blFile+'/'+str(j)+'_crop'+str(h*5+w)
                    
                    #blName= 'blur_processed_0_0_0'
                    #shName= 'shar_processed_0_0_0'
                    np.save(blName, blcrop)   
            
    return Rm, Gm, Bm, Rs, Gs, Bs

if __name__ == '__main__':
    Rm, Gm, Bm, Rs, Gs, Bs= normalize_save(30, 'test')
    print('Rmean= '+str(Rm))
    print('Gmean= '+str(Gm))
    print('Bmean= '+str(Bm))
    print('Rstd= '+str(Rs))
    print('Gstd= '+str(Gs))
    print('Bstd= '+str(Bs))