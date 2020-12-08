from keras.layers import Input,Dense,Reshape,Flatten,Dropout,BatchNormalization,Activation,ZeroPadding2D,Lambda,Add,Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import RMSprop
import keras as K
import matplotlib.pyplot as plt

import tensorflow as tf

import sys

import numpy as np


class GAN:
    def __init__(self,hwc=(224,224,3),latent_dim=20,lr=1e-4):
        self.hwc=hwc
        input_img=Input(shape=self.hwc)
        input_noise=Input(shape=(latent_dim,))
        
        optimizer=RMSprop(lr)
        
        self.discriminator=self.D_builder()
        self.discriminator.compile(optimizer=optimizer,loss='binary_crossentropy')
        
        self.discriminator.trainable=False
        self.VAE=self.VAE_builder(latent_dim)
        
        gen_img=self.VAE([input_img,input_noise])
        validity=self.discriminator(gen_img)
        
        self.combined=Model(inputs=[input_img,input_noise],outputs=validity)
        self.combined.compile(optimizer=optimizer,loss='binary_crossentropy')
        
    
    def VAE_loss(self,gen_img,label_img,mu,log_var):
        # mse + KL_loss
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        mse=tf.square(tf.add(gen_img,-label_img))
        KL_loss=tf.reduce_mean(-0.5*tf.reduce_sum(1+log_var-tf.square(mu)-tf.exp(log_var),axis=1),axis=0)
    
    def D_builder(self):
        # build a simple discriminator for AnoGan
        D_input_image=Input(shape=self.hwc)
        conv1=Conv2D(filters=64,kernel_size=3,strides=2)(D_input_image)
        conv1=LeakyReLU()(conv1)
        conv1=BatchNormalization()(conv1)
        
        conv2=Conv2D(filters=64,kernel_size=3,strides=2)(conv1)
        conv2=LeakyReLU()(conv2)
        conv2=BatchNormalization()(conv2)
        
        conv3=Conv2D(filters=64,kernel_size=3,strides=2)(conv2)
        conv3=LeakyReLU()(conv3)
        conv3=BatchNormalization()(conv3)
        
        conv4=Conv2D(filters=128,kernel_size=3,strides=2)(conv3)
        conv4=LeakyReLU()(conv4)
        conv4=BatchNormalization()(conv4)
        
        conv5=Conv2D(filters=256,kernel_size=3,strides=2)(conv4)
        conv5=LeakyReLU()(conv5)
        conv5=BatchNormalization()(conv5)
        
        flat=Flatten()(conv5)
        fc=Dense(1)(flat)
        
        return Model(D_input_image,fc)
    
    def coding_op(self,tensorlist):
        # noise,log_var,mu
        
        return Add()([Multiply()([tensorlist[0],K.backend.exp(0.5*tensorlist[1])]),tensorlist[2]])
    
    def VAE_builder(self,latent_dim):
        # build a VAE as generator
        input_img=Input(shape=self.hwc)
        
        conv1=Conv2D(filters=64,kernel_size=3,strides=2,padding='same')(input_img)
        conv1=LeakyReLU()(conv1)
        conv1=BatchNormalization()(conv1)
        
        conv2=Conv2D(filters=64,kernel_size=3,strides=2,padding='same')(conv1)
        conv2=LeakyReLU()(conv2)
        conv2=BatchNormalization()(conv2)
        
        conv3=Conv2D(filters=32,kernel_size=3,strides=2,padding='same')(conv2)
        conv3=LeakyReLU()(conv3)
        conv3=BatchNormalization()(conv3)
        
        conv4=Conv2D(filters=32,kernel_size=3,strides=2,padding='same')(conv3)
        conv4=LeakyReLU()(conv4)
        conv4=BatchNormalization()(conv4)
        
        encoder_shape=conv4.shape
        
        # here the latent variables are in shape [batch, 2*latent_dim]
        conv4=Flatten()(conv4)
        fc=Dense(2*latent_dim)(conv4)
        
        
        mu, log_var = Lambda(lambda x : [x[:,latent_dim:],x[:,:latent_dim]])(fc)
        noise=Input(shape=(latent_dim,))
        
        
        code=Lambda(self.coding_op)([noise,log_var,mu])
        
        
        de_input=Dense(int(encoder_shape[-3]) * int(encoder_shape[-2]) * 16)(code)
        de_input=Reshape([int(encoder_shape[-3]), int(encoder_shape[-2]), 16])(de_input)
        
        
        de_conv1_1=Conv2D(32,3,padding='same')(de_input)
        de_conv1_1=BatchNormalization()(de_conv1_1)
        de_conv1_1=LeakyReLU(alpha=0.2,name='de_conv1_1')(de_conv1_1)
        de_conv1_1=UpSampling2D(2)(de_conv1_1)
        
        de_conv1_2=Conv2D(32,3,padding='same')(de_conv1_1)
        de_conv1_2=BatchNormalization()(de_conv1_2)
        de_conv1_2=LeakyReLU(alpha=0.2,name='de_conv1_2')(de_conv1_2)
        de_conv1_2=UpSampling2D(2)(de_conv1_2)
        
        de_conv2_1=Conv2D(64,3,padding='same')(de_conv1_2)
        de_conv2_1=BatchNormalization()(de_conv2_1)
        de_conv2_1=LeakyReLU(alpha=0.2,name='de_conv2_1')(de_conv2_1)
        de_conv2_1=UpSampling2D(2)(de_conv2_1)
        
        de_conv2_2=Conv2D(64,3,padding='same')(de_conv2_1)
        de_conv2_2=BatchNormalization()(de_conv2_2)
        de_conv2_2=LeakyReLU(alpha=0.2,name='de_conv2_2')(de_conv2_2)
        de_conv2_2=UpSampling2D(2)(de_conv2_2)
        
        de_outputs=Conv2D(3,3,padding='same',activation='tanh')(de_conv2_2)
        
        
        return Model(inputs=[input_img,noise],outputs=de_outputs)
    
    
    
    
    