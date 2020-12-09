import numpy as np
import cv2
import pickle
import os

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