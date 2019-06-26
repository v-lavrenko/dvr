#!/usr/local/bin/python3

import numpy as np
import cv2, sys, os, copy, re, time, bz2, h5py

from keras.models import Model, Sequential, load_model
from keras.layers import InputLayer, Dense, Dropout, Reshape #, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, SpatialDropout2D 
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, SpatialDropout3D 
from keras import backend
from keras.losses import binary_crossentropy
#import matplotlib.pyplot as plt
#from keras.constraints import max_norm

def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

def waitkey(ms): return chr (cv2.waitKey(ms) & 0xFF)

def pad(n,b): return (1+(n>>b))<<b
def nextpow2(n): return 1<<int(1+np.log2(n-1))

def parse_ROI(s): return None if s is None else tuple(map(int,re.split('[x+]',s)))

def getROI(img,prompt='Select Region of Interest'):
    cv2.imshow('vid', img)
    x,y,w,h = cv2.selectROI(prompt, img, fromCenter=False, showCrosshair=True)
    # cv2.destroyAllWindows() # doesn't work even with cv2.waitKey(1)
    ww,hh = pad(w,5),pad(h,5) # nextpow2(w),nextpow2(h)
    dw,dh = (ww-w)<<1,(hh-h)<<1
    return x-dw,y-dw,ww,hh

def visualize(imgs,wait=1,tag='haha'):
    global ovid, opath
    font = cv2.FONT_HERSHEY_PLAIN
    #imgs = [cv2.resize(x.astype('uint8'), (320,180)) for x in imgs]
    row1 = np.concatenate(imgs[0:2],axis=1)
    row2 = np.concatenate(imgs[2:4],axis=1)
    img = np.concatenate([row1,row2],axis=0)
    #img = np.concatenate(imgs,axis=1)
    cv2.putText(img, tag, (5,15), font, 1, (0,0,0), 3)
    cv2.putText(img, tag, (5,15), font, 1, (255,255,255), 1)
    cv2.imshow('xxx', img)
    waitkey(wait)
    if opath and not ovid:
        codec = 'XVID'
        fcc = cv2.VideoWriter_fourcc(*codec)
        ovid = cv2.VideoWriter(opath, fcc, 5, img.shape[:2], isColor=True)
    if opath: ovid.write(img)

def vid_data(xpath,ypath,thr=20000,show=False,c3d=False):
    xcap,ximg = cv2.VideoCapture(xpath),None
    ycap,yimg = cv2.VideoCapture(ypath),None
    X,Y,frame = [],[],0
    f1,f2,f3 = None,None,None # frames at time t-1, t-2, t-3
    ema9,ema99 = None,None

    while True:
        xret, ximg = xcap.read()
        yret, yimg = ycap.read()
        frame += 1
        if not xret or not yret: break # end of stream
        if yimg.sum() < thr: continue # skip empty target
        if roi is not None:
            x,y,w,h = roi
            ximg = ximg[y:y+h,x:x+w]
            yimg = yimg[y:y+h,x:x+w]
        # if c3d:
        #     if frame == 1: ema9,ema99 = ximg,ximg
        #     ema90 = 0.90 * ema90 + 0.10 * ximg
        #     ema99 = 0.99 * ema99 + 0.01 * ximg
        #     ximg = np.array([ximg,ema90,ema99])
        if f1 is None: f1,f2 = copy.copy(ximg),copy.copy(ximg)
        f1,f2,f3 = ximg,f1,f2 # TODO: replace with [prev,this,next]
        if c3d: ximg = np.array([f1,f2,f3])
        X.append(ximg)
        Y.append(yimg)
        if c3d: visualize((f1,np.abs(f1-f2),np.abs(f1-f3),yimg),tag=('training frame %d'%frame))
        else:   visualize((ximg,yimg,yimg,ximg),tag=('training frame %d'%frame))
        print ('%d %.0f'%(frame,yimg.sum()), ximg.shape)

    xcap.release(), ycap.release()
    return np.array(X),np.array(Y)
    #X = np.moveaxis (X,-1,1) # channels -> 2nd 
    #Y = np.moveaxis (Y,-1,1) # channels -> 2nd

def poly_expand(X,cross=False): # polynomial expansion of rows in X
    if cross: X = [np.outer(x,x).flatten() for x in X]
    else: X = [np.concatenate((x,x**2),axis=1) for x in X]
    return np.array(X)

def stratify_pixels(X,Y,ratio=1):
    if ratio <= 0: return X,Y
    sums = Y.sum(axis=1)
    pos = np.where(sums>1)[0] # positives: non-black pixels
    neg = np.where(sums<=1)[0] # negatives: black pixels
    neg = np.random.choice(neg,ratio*len(pos)) # balance
    if verbose: print ('stratify: %d/%d pos/neg'%(len(pos),len(neg)))
    idx = np.append(pos,neg)
    return X[idx],Y[idx]

def model_pixels(deep=0,wide=9):
    model = Sequential()                     # 3 x 720 x 1280
    model.add(InputLayer((3,)))
    for i in range(deep): model.add(Dense(wide,activation='relu'))
    model.add(Dense(3,activation='sigmoid'))
    return model

def fit_pixels(X,Y,stratify=0,mpath=None):
    n,r,c,ch = X.shape
    X,Y = X.reshape((n*r*c,ch)),Y.reshape((n*r*c,ch)) # frames -> pixels
    model = model_pixels(deep=5)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
    for i in range(1,201):
        XX,YY = stratify_pixels(X,Y,stratify)
        batch,epochs = int(np.sqrt(10*len(XX))),i #int(len(X)/len(XX))
        h = model.fit(XX,YY,epochs=epochs,batch_size=batch,shuffle=True,verbose=0)
        if mpath: model.save(mpath)
        loss,mae = h.history['loss'][0],255*h.history['mean_absolute_error'][0]
        #tag = "epoch: %.0f loss: %.4f mae: %.2f %s %s" % ((i*i+i)/2, loss, mae, mpath, xpath)
        tag = "epoch: %.0f loss: %.4f mae: %.2f %s %s" % (i, loss, mae, mpath, xpath)
        print(tag)
        if verbose:
            P = model.predict(X,batch_size=batch,verbose=1)
            _X = 255*X.reshape((n,r,c,ch))
            _Y = 255*Y.reshape((n,r,c,ch))
            _P = 255*P.reshape((n,r,c,ch))
            _D = np.abs(_P-_Y)
            it = np.random.choice(n)
            visualize((_X[it],_P[it],_Y[it],_D[it]),tag=tag)

def sparse_CE(Y,P): # NaN = missing target => skip backprop
    return 0 if np.isnan(P).any() else binary_crossentropy(Y,P)
    #return 0 if P.sum == 0 else binary_crossentropy(Y,P)
    #return 0 if P.nonzero().any() else binary_crossentropy(Y,P)

def model_cnn0(shape=(1280,720,3)):
    m = Sequential()
    m.add(InputLayer(shape))
    m.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
    m.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
    m.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
    m.add(Convolution2D(3, (3, 3), activation='sigmoid', padding='same'))
    return m

def model_cnn1(shape=(1280,720,3)):
    m = Sequential()
    m.add(InputLayer(shape))
    m.add(Convolution2D(8, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.1))
    m.add(MaxPooling2D(pool_size=(2,2))) # 1/2
    m.add(Convolution2D(16, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.2))
    m.add(MaxPooling2D(pool_size=(2,2))) # 1/4
    m.add(Convolution2D(32, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.5))
    m.add(MaxPooling2D(pool_size=(2,2))) # 1/8
    m.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.5))
    m.add(MaxPooling2D(pool_size=(2,2))) # 1/16
    m.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.5))
    m.add(UpSampling2D((2, 2)))          # 1/8
    m.add(Convolution2D(32, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.5))
    m.add(UpSampling2D((2, 2)))          # 1/4
    m.add(Convolution2D(16, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.2))
    m.add(UpSampling2D((2, 2)))          # 1/2
    m.add(Convolution2D(8, (3,3), activation='relu', padding='same'))
    #m.add(SpatialDropout2D(0.1))
    m.add(UpSampling2D((2, 2)))          # 1/1
    m.add(Convolution2D(3, (3,3), activation='sigmoid', padding='same'))
    return m

def model_c3d1(shape):
    m = Sequential()
    m.add(InputLayer(shape))               # (3, 256, 256) 3ch
    m.add(Convolution3D(9, (3,3,3), activation='relu', padding='same'))
    m.add(SpatialDropout3D(0.1))
    m.add(MaxPooling3D(pool_size=(1,2,2))) # (3, 128, 128) 
    m.add(Convolution3D(9, (3,3,3), activation='relu', padding='same'))
    m.add(SpatialDropout3D(0.5))
    m.add(MaxPooling3D(pool_size=(1,2,2))) # (3, 64, 64) 
    m.add(Convolution3D(9, (3,3,3), activation='relu', padding='same'))
    m.add(SpatialDropout3D(0.5))
    m.add(MaxPooling3D(pool_size=(3,2,2))) # (1, 32, 32) 
    m.add(Convolution3D(9, (3,3,3), activation='relu', padding='same'))
    m.add(SpatialDropout3D(0.5))
    m.add(UpSampling3D(     size=(1,2,2))) # (1, 64, 64)
    m.add(Convolution3D(9, (3,3,3), activation='relu', padding='same'))
    m.add(UpSampling3D(     size=(1,2,2))) # (1, 128, 128)
    m.add(Convolution3D(9, (3,3,3), activation='relu', padding='same'))
    m.add(UpSampling3D(     size=(1,2,2))) # (1, 256, 256)
    m.add(Convolution3D(3, (3,3,3), activation='sigmoid', padding='same'))
    m.add(Reshape((256,256,3))) # ERROR: total size must be unchanged    
    #m.add(MaxPooling3D(pool_size=(3,1,1))) # (1, 256, 256, 3)
    #m.add(Convolution3D(3, (3,3,3), activation='relu', padding='same'))
    return m

def fit_cnn(X,Y,mpath=None):
    model = model_cnn0(shape=X[0].shape)
    #model = model_c3d1(shape=X[0].shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
    batch = int(np.sqrt(len(X)))
    for i in range(1,500):
        h = model.fit(X,Y,epochs=1,batch_size=batch,shuffle=True,verbose=1)
        if mpath: model.save(mpath)
        loss,mae = h.history['loss'][0],255*h.history['mean_absolute_error'][0]
        tag = "epoch: %.0f loss: %.4f mae: %.2f %s %s" % (i, loss, mae, mpath, xpath)
        print(tag)
        if verbose:
            P = model.predict(X,batch_size=len(X),verbose=1)
            D = np.abs(P-Y)
            ex = 1 # np.random.choice(len(X))
            _X = (255*X[ex]).astype('uint8')
            _P = (255*P[ex]).astype('uint8')
            _Y = (255*Y[ex]).astype('uint8')
            _D = (255*D[ex]).astype('uint8')
            visualize((_X,_P,_Y,_D),tag=tag)
            
verbose = argp('-v')
roi = parse_ROI(getarg('-roi',None))
stratify = int(getarg('-stratify','0'))
xpath,ypath = getarg('-x',None),getarg('-y',None)
opath,mpath = getarg('-o',None),getarg('-m',None)
ovid = None

if not xpath or not ypath:
    sys.stderr.write('Usage: detrain.py -x source.avi -y target.avi\n')
    sys.exit(1)

X,Y = vid_data(xpath,ypath,c3d=False)
X,Y = X.astype('float32')/255, Y.astype('float32')/255 # [0..1]
backend.set_image_dim_ordering('tf') # tf:(R,C,ch), th:(ch,R,C)
#fit_pixels(X,Y,stratify=stratify,mpath=mpath)
fit_cnn(X,Y,mpath=mpath)

if ovid: ovid.release()
cv2.destroyAllWindows()
print('done')

#./denoise.py -in d.avi -train 75 -mask -smooth 1 -thresh 1 -scale 2 -blur 21 -save
#./detrain.py -y anno.d.avi -x denoised.d.avi

# https://github.com/harvitronix/five-video-classification-methods/blob/master/models.py
# https://stackoverflow.com/questions/42633644/using-keras-for-video-prediction-time-series
# https://github.com/keras-team/keras/issues/369
