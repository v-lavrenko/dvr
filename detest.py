#!/usr/local/bin/python3

import numpy as np
import cv2, sys, os, copy, re, time, bz2, h5py

def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

def waitkey(ms): return chr (cv2.waitKey(ms) & 0xFF)

def parse_ROI(s): return None if s is None else tuple(map(int,re.split('[x+]',s)))

def visualize(imgs,wait=1,tag='haha'):
    global ovid
    #imgs = [cv2.resize(x.astype('uint8'), (320,180)) for x in imgs]
    img = np.concatenate(imgs,axis=1)
    cv2.imshow(tag, img)
    waitkey(wait)

verbose = True #argp('-v')
ipath,opath,mpath = getarg('-i',None),getarg('-o',None),getarg('-m',None)

if not ipath or not opath or not mpath:
    sys.stderr.write('Usage: detest.py -i source.avi -m model.h5 -o result.avi\n')
    sys.exit(1)

roi = parse_ROI(getarg('-roi',None))    
pix,cnn = argp('-pix'),argp('-cnn')

from keras.models import Model, Sequential, load_model
from keras import backend
backend.set_image_dim_ordering('tf') # tf:(R,C,ch), th:(ch,R,C)
model = load_model(mpath)

cap = cv2.VideoCapture(ipath)
fps = int(getarg('-fps','10'))
codec = getarg('-codec','XVID') # X264 H264 AVC1 XVID MJPG
fcc = cv2.VideoWriter_fourcc(*codec)
ovid = None
frame = 0

while True:
    ret, img = cap.read()
    if not ret: break
    frame += 1
    if roi is not None:
        x,y,w,h = roi
        img = img[y:y+h,x:x+w]
    r,c,ch = img.shape
    if pix:
        X = img.reshape((r*c,ch)).astype('float32')/255
        Y = model.predict(X,batch_size=(r*c),verbose=verbose)
        omg = (255*Y.reshape(r,c,ch)).astype('uint8')
    else:
        X = img.reshape((1,r,c,ch)).astype('float32')/255
        Y = model.predict(X,batch_size=1,verbose=verbose)
        omg = (255*Y[0]).astype('uint8')
    if verbose: visualize((img,omg),wait=1)
    if ovid is None:
        ovid = cv2.VideoWriter(opath, fcc, fps, omg.shape[:2], isColor=True)
    ovid.write(omg)
    if frame == 156 or frame == 878:
        fpath = 'xxx.'+str(frame)+'.png'
        cv2.imwrite(fpath,omg)
        print (fpath,omg.shape)

cap.release()
if ovid: ovid.release()
cv2.destroyAllWindows()
