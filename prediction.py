import keras as K
import numpy as np
import cv2
import os
import data_loader as dl
from model import GAN, coding_op
import math
import utils
import re
import pickle
import tensorflow as tf
# system settings
os.environ['CUDA_VISIBLE_DEVICES']='0'

def val(img_dir,model,hwc,latent_dim,dst_dir='./val_results'):
    # the model should be a keras model, inputs:[img,noise];outputs:[gen_img,mu,log_var]
    if not os.path.exists(dst_dir):
        os.system('mkdir '+str(dst_dir))
    loader1=dl.data_loader(img_dir,hwc,train_val=0,add_val=img_dir)
    for b in range(len(loader1.val_names)):
        b_img=loader1.get_val(index=(b,b+1))
        #input_noise=np.random.normal(size=(1,int(224/16),int(224/16),latent_dim))
        input_noise=np.random.normal(size=(1,latent_dim))
        gen_img,mu,log_var=model.predict([b_img,input_noise])
        
        save_img=loader1.postprocess_img(gen_img[0])
        cv2.imwrite('result'+str(b)+'.png',save_img)
    print('results saved')
    
    
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

def VAE_loss(gen_img,label_img,mu,log_var,KL_ratio=0.1):
    # mse + KL_loss
    mse=tf.reduce_mean(tf.square(tf.add(gen_img,-label_img)))
    KL_loss=tf.reduce_mean(-0.5*tf.reduce_sum(1+log_var-tf.square(mu)-tf.exp(log_var),axis=1),axis=0)
    return mse+KL_ratio*KL_loss


VAE=K.models.load_model('G1.h5',custom_objects={'VAE_loss':VAE_loss,'coding_op':coding_op,'tf':tf})
VAE.summary()
print('model loaded!')
val('.\\val_samples\\',VAE,hwc=(112,112,3),latent_dim=128)
