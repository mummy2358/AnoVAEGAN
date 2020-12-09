import keras as K
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import cv2
import os
import data_loader as dl
import model
import math
import pickle

# system settings
os.environ['CUDA_VISIBLE_DEVICES']='0'
'''
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''

def encode(ls1):
    # can change encoding algorithm here
    # encoding algorithm should be invertible and in [0,255]
    ls1=np.array(ls1)
    ls2=list(255-ls1)
    return ls2

def decode(ls2):
    ls2=np.array(ls2)
    ls1=list(255-ls2)
    return ls1

def pickle_save(model,filename):
    # save model object to file with pickle dump
    out_put = open(filename, 'wb')
    obj = pickle.dumps(model)
    out_put.write(obj)
    out_put.close()

def pickle_load(filename):
    # return file as bytes
    with open(filename, 'rb') as file:
        return file.read()
        
def final_save(model,filename):
    pickle_save(model,'dur.inv')
    fbytes=pickle_load('dur.inv')
    os.system('DEL dur.inv')
    fls_encoded=encode(list(fbytes))
    obj=bytes(fls_encoded)
    out_put = open(filename, 'wb')
    out_put.write(obj)
    out_put.close()

def final_load(filename):
    fbytes=pickle_load(filename)
    fls=decode(list(fbytes))
    fbytes_decoded=bytes(fls)
    model = pickle.loads(fbytes_decoded)
    return model

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
            final_save(gan0,filename)
            print('saving loss to '+log_file+' ...')
            final_save(loss_tracking,log_file)
    
    
    
    
    
    
    
    