import numpy as np
import keras as K
import cv2
import os
import sys
import re
import time
from keras.preprocessing.image import ImageDataGenerator

class data_loader:
    def __init__(self,root='./',hwc=(224,224,3),train_val=0.9,add_train=None,add_val=None,reweight_map=None):
        # directory discription: all images including labels are in the same dir
        # the appendix: 0model.bmp, 1check.bmp, 2label.bmp
        # hwc: the size of the given image
        # train_val: ratio between the base train and val
        # add_train: additional training directory
        # add_val: additional validation directory
        # reweight_map: {'':,'':,'':} for each sample, if the file directory include a key, the sample is multiplied by the factor mapped from the key
        
        self.root=root
        self.hwc=hwc
        self.fullnamelist=[]
        self.traversal_directory(self.root)
        
        namelist=[]
        for name in self.fullnamelist:
            if re.search('.bmp',name):
                namelist.append(name[:-4])
        
        np.random.seed(0)
        np.random.shuffle(namelist)
        self.train_val=train_val
        self.train_names=namelist[:int(train_val*len(namelist))]
        self.val_names=namelist[int(train_val*len(namelist)):]
        
        if add_train:
            self.fullnamelist=[]
            self.traversal_directory(add_train)
            addtrain_namelist=[]
            for name in self.fullnamelist:
                if re.search('.bmp',name):
                    addtrain_namelist.append(name[:-4])
            self.train_names+=addtrain_namelist
        
        if add_val:
            self.fullnamelist=[]
            self.traversal_directory(add_val)
            addval_namelist=[]
            for name in self.fullnamelist:
                if re.search('.bmp',name):
                    addval_namelist.append(name[:-4])
            self.val_names+=addval_namelist
        
        print('training number:'+str(len(self.train_names)))
        print('validation number:'+str(len(self.val_names)))
        
        self.indices=list(range(len(self.train_names)))
        self.sample_counter=0
        self.reweight_map=reweight_map
    
    def transform(self,batch_img,seed=0):
        aug=K.preprocessing.image.ImageDataGenerator(height_shift_range=3,width_shift_range=3,shear_range=3.0,rotation_range=5)
        batch_size=np.shape(batch_img)[0]
        aug_iter=aug.flow(x=batch_img,batch_size=batch_size,shuffle=False,seed=seed)
        return next(aug_iter)
    
    def transform_total(self,batch_img,seed=0):
        aug=K.preprocessing.image.ImageDataGenerator(height_shift_range=30,width_shift_range=30,shear_range=3.0,rotation_range=45,horizontal_flip=True,vertical_flip=True,brightness_range=(0.95,1.05))
        batch_size=np.shape(batch_img)[0]
        aug_iter=aug.flow(x=batch_img,batch_size=batch_size,shuffle=False,seed=seed)
        return next(aug_iter)
    
    def rect_dropout(self,img,maxratioh=0.25,maxratiow=0.25):
        # maxh and maxw are the maximum ratios that a rectangle out of the whole image
        hwc=np.shape(img)
        maxh=int(maxratioh*hwc[0])
        maxw=int(maxratiow*hwc[1])
        h=np.random.randint(1,maxh)
        w=np.random.randint(1,maxw)
        r=np.random.randint(0,hwc[0]-h-1)
        c=np.random.randint(0,hwc[1]-w-1)
        mask=np.ones([*hwc[:-1],1])
        mask[r:r+h,c:c+w]=[0]
        res=mask*img
        
        res=np.clip(res,0,255)
        res=res.astype(np.uint8)
        return res
    
    def preprocess_img(self,img):
        # for gan specifically, the input img should be ranged to [-1,1]
        img=np.array(img)
        img=img.astype(np.float32)
        img=cv2.resize(img,(self.hwc[:-1][1],self.hwc[:-1][0]))
        img=img/255.0
        img=2*(img-0.5)
        return img
    
    def preprocess_label(self,labelimg):
        # here we suppose class 0 is background
        labelimg=np.array(labelimg)
        labelimg=labelimg.astype(np.float32)
        labelimg=cv2.resize(labelimg,(self.hwc[:-1][1],self.hwc[:-1][0]))
        mask=labelimg[:,:,0]<0.5
        res=np.ones([*self.hwc[:-1],2])
        res[:,:,0]=res[:,:,0]*mask
        res[:,:,1]=res[:,:,1]*np.logical_not(mask)
        return res
    
    def postprocess_img(self,img):
        # inverse process of preprocessing
        img=np.array(img)
        img=(img+1)*127.5
        img=np.clip(img,0,255)
        img=img.astype(np.uint8)
        return img
    
    def postprocess_label(self,pred,T=None):
        # input: [h,w,2]
        # return argmax mask img in shape [h,w,3]
        pred=np.array(pred)
        if T:
            pred=utils.threshold(pred,T)
        mask=pred[:,:,1]
        mask=mask*255
        labelimg=np.clip(mask,0,255)
        labelimg=labelimg.astype(np.uint8)
        return labelimg
    
    def traversal_directory(self,root,filetype='.bmp'):
        names=os.listdir(root)
        for name in names:
            filedir=os.path.join(root,name)
            if os.path.isdir(filedir):
                self.traversal_directory(filedir)
            elif re.search(filetype,name):
                self.fullnamelist.append(filedir)
    
    def train_next_batch(self,batch_size=10):
        batch_inputs=[]
        batch_outputs=[]
        batch_reweight=[]
        while True:
            name_prefix=self.train_names[self.indices[self.sample_counter]]
            testimg=cv2.imread(name_prefix+'.bmp')
            batch_reweight.append(1.0)
            if self.reweight_map:
                for key in self.reweight_map:
                    if re.search(key,name_prefix):
                        batch_reweight[-1]=batch_reweight[-1]*self.reweight_map[key]
            
            inputs=self.rect_dropout(testimg,0.4,0.4)
            outputs=testimg
            
            seed=np.random.randint(low=0,high=65535)
            
            inputs=self.transform(np.array([inputs]),seed=seed)[0]
            inputs=self.preprocess_img(inputs)
            batch_inputs.append(inputs)
            
            outputs=self.transform(np.array([outputs]),seed=seed)[0]
            outputs=self.preprocess_img(outputs)
            batch_outputs.append(outputs)
            
            self.sample_counter+=1
            if self.sample_counter%batch_size==0 or self.sample_counter==len(self.train_names):
                yield np.array(batch_inputs),np.array(batch_outputs),np.array(batch_reweight)
                batch_inputs=[]
                batch_outputs=[]
                batch_reweight=[]
            if self.sample_counter==len(self.train_names):
                np.random.shuffle(self.indices)
                self.sample_counter=0
    
    def get_val(self,index=None):
        # index should be two int numbers
        batch_testimg=[]
        if not index:
            index=[0,len(self.val_names)]
        for i in range(*index):
            testimg=cv2.imread(self.val_names[i]+'.bmp')
            
            testimg=self.preprocess_img(testimg)
            
            batch_testimg.append(testimg)
            
        return np.array(batch_testimg)
