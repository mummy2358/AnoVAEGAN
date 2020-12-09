import keras as K
import numpy as np
import cv2
import os
import data_loader as dl
import model
import math
import pickle
import utils
# system settings
os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__=='__main__':
    latent_dim=20
    batch_size=2
    max_epoch=100
    save_epoch=10
    
    filename='VAE_gan.model'
    loader0=dl.data_loader(root='simple_pool_hmw2',hwc=(112,112,3))
    iterator0=loader0.train_next_batch(batch_size)
    
    log_file='losses.model'
    
    gan0=model.GAN(hwc=(112,112,3),latent_dim=latent_dim,lr=1e-4)
    
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
            
            loss_G = gan0.combined.train_on_batch(x={'input_img':b_inputs,'input_noise':b_noise,'label_img':b_outputs},y=None)
            
            loss_D_epoch+=loss_D
            loss_G_epoch+=loss_G
            
        loss_D_epoch=loss_D_epoch/math.ceil(len(loader0.train_names)/batch_size)
        loss_G_epoch=loss_G_epoch/math.ceil(len(loader0.train_names)/batch_size)
        print('epoch '+str(e+1)+': D '+str(loss_D_epoch)+'    G '+str(loss_G_epoch))
        
        loss_tracking['loss_D'].append(loss_D_epoch)
        loss_tracking['loss_G'].append(loss_G_epoch)
        
        if (e+1)%save_epoch==0:
            print('saveing model to '+filename+' ...')
            utils.final_save(gan0,filename)
            print('saving loss to '+log_file+' ...')
            utils.final_save(loss_tracking,log_file)
    
    