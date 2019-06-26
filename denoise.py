#!/usr/local/bin/python

import numpy as np
import cv2, sys, os
from math import erf
#from matplotlib import pyplot as plt

def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

def die():
    if save: vid.release()
    cap.release()
    cv2.destroyAllWindows()
    print 'result:', opath

def cap_size(cap): return (int(cap.get(3)), int(cap.get(4)))

def waitkey(ms): return chr (cv2.waitKey(ms) & 0xFF)

import signal
signal.signal(signal.SIGINT, die)
signal.signal(signal.SIGTERM, die)

ipath = getarg('-in',None)
if len(sys.argv) < 2 or not ipath:
    sys.stderr.write('Usage: denoise.py [-in in.avi]\n')
    sys.exit(1)

scale = float(getarg('-scale','0'))
thresh = float(getarg('-thresh','3'))
smooth = float(getarg('-smooth','1'))
train = int(getarg('-train','10'))
truth = getarg('-truth',None)
dump  = set((int(f) for f in getarg('-dump','0').split(',')))
opath = getarg('-o','denoised.'+ipath)
codec = getarg('-codec','XVID') # X264 H264 AVC1 XVID MJPG
blur  = int(getarg('-blur','1'))
gray,rgb,hsv = argp('-gray'),argp('-rgb'),argp('-hsv')
both,orig = argp('-both'),argp('-orig')
mask,heat = argp('-mask'),argp('-heat')
amp = float(getarg('-amp','0'))
_abs = argp('-abs')
pause = False
x1,x2 = argp('-1x'),argp('-2x')

save,show = argp('-save'),argp('-show')

frame,mean,mean2,stdev = 0,None,None,None

cap = cv2.VideoCapture(ipath)

if truth:
    timage = cv2.imread(truth.split(':')[1],0)
    tframe = int(truth.split(':')[0])
    print tframe, timage.shape

if save:
    fcc = cv2.VideoWriter_fourcc(*codec)
    vid = cv2.VideoWriter(opath, fcc, 10, cap_size(cap), isColor=(not gray))

while True:
    key = chr (cv2.waitKey(1) & 0xFF)
    if key == 'q': break # quit
    if key == 'r': pass # implement rewind
    if key == 'x':
        fpath = ipath+'.'+str(frame)+'.png'
        cv2.imwrite(fpath,img)
        print fpath
    if key == ' ': pause = not pause
    if pause: continue
    
    ret, img = cap.read()
    if not ret: break
    frame += 1
    if x2: img = cv2.resize(img, (640,360))
    if x1: img = cv2.resize(img, (320,180))
    
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif rgb:
        I = cv2.resize(img, (320,180))
        (B,G,R) = cv2.split(I)
        if True:
            O = np.zeros(B.shape).astype('uint8')
            R = cv2.merge([O, O, R])
	    G = cv2.merge([O, G, O])
	    B = cv2.merge([B, O, O])
        img = np.concatenate((R,G,B),axis=0)
    elif hsv:
        I = cv2.resize(img, (320,180))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
        (H,S,V) = cv2.split(I)
        img = np.concatenate((H,S,V),axis=0)
        
    #if blur > 1: img = cv2.GaussianBlur(img,(blur,blur),0)

    X = img.astype(float)
    #if blur > 1: X = cv2.GaussianBlur(X,(blur,blur),0)
    if frame == 1:      mean,mean2 = X, X*X
    elif frame < train: mean,mean2 = mean+X, mean2+X*X
    elif frame == train:
        mean  /= train
        mean2 /= train
        stdev = np.sqrt(mean2 - mean*mean) + smooth
        #print 'mean1:', mean.shape, mean.min(), mean.sum() / mean.size, mean.max()
        #print 'mean2:', mean2.shape, mean2.min(), mean2.sum() / mean2.size, mean2.max()
        #print 'stdev:', stdev.shape, stdev.min(), stdev.sum() / stdev.size, stdev.max()
    elif frame > train:
        I = X
        #for iter in range(2):
        Z = (I - mean) / stdev        
        if blur > 1:  Z = cv2.GaussianBlur(Z,(blur,blur),0)
        if _abs:      Z = np.abs(Z)
        if scale > 0: I = (X if mask else 255) / (1 + np.exp(-scale * (Z-thresh)))
        else:         I = (X if mask else 255) * (Z > thresh)
        if amp > 0:   I = I * amp
        if heat and not gray:
            P = 1 / (1 + np.exp(-scale * (Z-thresh)))
            if len(P.shape) == 3: P = P.mean(axis=2)
            R = 255 * (P > 0.60) * P
            B = 255 * (P < 0.40) * (1-P)
            G = np.zeros(P.shape).astype('uint8')
            I = cv2.merge([B.astype('uint8'), G, R.astype('uint8')])
        
        if show: print '%3d X: [%.0f %.0f %.0f] Z:[%.2f %.2f %.2f]' % (frame, X.min(), X.mean(), X.max(), Z.min(), Z.mean(), Z.max())
    if truth and frame == tframe:
        print timage.shape, I.shape
        if len(I.shape) == 3: I = I.max(axis=2)
        if timage.shape != I.shape: timage = cv2.resize(timage, I.shape[::-1])
        diff = (I - timage + 255) / 2
        cv2.imshow('diff',diff.astype('uint8'))
        cv2.waitKey(0)

        
    R = I if frame > train else mean
    R = R.astype('uint8')
    OUT = img if orig else np.concatenate((img,R),axis=1) if both else R
        #if gray or rgb or hsv: LR = cv2.applyColorMap(LR, cv2.COLORMAP_HOT)
    if show:
        cv2.imshow('result',OUT)
    if save:
        vid.write(OUT)
    if frame in dump:
        dir = 'denoise/'+str(frame)
        fpath = dir+'/'+str(frame)+'_'.join(sys.argv[1:])+'.png'
        print fpath
        try: os.makedirs(dir)
        except: pass
        cv2.imwrite(fpath,OUT)

    

die()
