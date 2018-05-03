from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2


import numpy as np
import os
import cv2


def get_patches(img,win_size = 32,win_step = 8):
    
    output = []
    h,w = img.shape[0],img.shape[1]
    
    for x in range(0,w-win_size+1,win_step):
    
        if x < w-h:
            patch = img[:,x:x+h]
            patch = cv2.resize(patch,(win_size,win_size))
            output.append(patch)
            
        for y in range(0,h-win_size+1,win_step):
            patch = img[y:y+win_size,x:x+win_size]
            output.append(patch)
            
    if w<win_size:
        
        for y in range(0,h-win_size+1,win_step):
            patch = cv2.resize(img[y:y+win_size,:],(win_size,win_size))
            output.append(patch)
            
    out = np.array(output).astype(np.float32)
    
    if len(out.shape)==3:
        out = np.expand_dims(out,-1)
        
    return out
    
    
def load_data(path,dataset):
    
    if dataset == 'CVSI':
        img = plt.imread(path)
        
    if dataset == 'SIW':
        img = cv2.imread(path,0)
        
    if dataset == 'MLE2E':
        img = cv2.imread(path)
    
    if dataset == 'ICDAR':
        img = plt.imread(path)
    
    img = (img)/255
    r,c = img.shape[0],img.shape[1]
    
    img = cv2.resize(img,(int(40*(c/r)),40)) # import __future__ 
    
    patches = get_patches(img)
    
    return np.expand_dims(patches.astype(np.float32),0)