import keras as K
import numpy as np
import cv2
import os
import data_loader as dl
from model import GAN
import math
import pickle
import utils
# system settings
os.environ['CUDA_VISIBLE_DEVICES']='1'

if __name__=='__main__':
    latent_dim=128
    batch_size=8
    max_epoch=10000
    save_epoch=10
    
    loader0=dl.data_loader(root='simple_pool_hmw2',hwc=(112,112,3))
    iterator0=loader0.train_next_batch(batch_size)
    
    log_file='losses.model'
    resume_G='G.h5'
    resume_D='D.h5'
    resume_combined='combined.h5'
    save_G='G1.h5'
    save_D='D1.h5'
    save_combined='combined1.h5'
    
    gan0=GAN(hwc=(112,112,3),latent_dim=latent_dim,lr=1e-4)
    G=gan0.VAE
    D=gan0.discriminator
    combined=gan0.combined
    
    
    loss_tracking={"loss_D":[],"loss_G":[]}
    for e in range(max_epoch):
        loss_D_epoch=0
        loss_G_epoch=0
        for b in range(math.ceil(len(loader0.train_names)/batch_size)):
            b_inputs,b_outputs,b_reweight=next(iterator0)
            numb=np.shape(b_inputs)[0]
            
            b_noise=np.random.normal(size=(numb,latent_dim))
            gen_img,mu,log_var=gan0.VAE.predict([b_inputs,b_noise])
            
            valid=np.ones(shape=(numb,1))
            fake=np.zeros(shape=(numb,1))
            
            loss_real=gan0.discriminator.train_on_batch(b_inputs,valid)
            loss_fake=gan0.discriminator.train_on_batch(gen_img,fake)
            loss_D = 0.5 * np.add(loss_real, loss_fake)
            
            loss_G = gan0.combined.train_on_batch(x={'input_img':b_inputs,'input_noise':b_noise,'label_img':b_outputs,'label_D':valid},y=None)
            
            loss_D_epoch+=loss_D
            loss_G_epoch+=loss_G
            
        loss_D_epoch=loss_D_epoch/math.ceil(len(loader0.train_names)/batch_size)
        loss_G_epoch=loss_G_epoch/math.ceil(len(loader0.train_names)/batch_size)
        print('epoch '+str(e+1)+': D '+str(loss_D_epoch)+'    G '+str(loss_G_epoch))
        
        loss_tracking['loss_D'].append(loss_D_epoch)
        loss_tracking['loss_G'].append(loss_G_epoch)
        
        if (e+1)%save_epoch==0:
            print('saveing model to '+save_G+'    '+save_D+'  ...')
            #gan0.save(filename)
            G.save(save_G)
            D.save(save_D)
            combined.save(save_combined)
            print('saving loss to '+log_file+' ...')
            #np.save(filename,loss_tracking)
            utils.final_save(loss_tracking,log_file)
    
    
