#!/usr/local/bin/python3

import numpy as np
import cv2, sys, os, copy, re#, time, tables, bz2, h5py

usage = '''vid.py -i in.avi -o out.avi [-v] [options]
            -dry ... dry-run: don't write out.avi
           -show ... show result in a window
   -size 640x480 ... resize input
        -fps 10  ... frames per second, default: 10
     -codec XVID ... X264 H264 AVC1 XVID MJPG
           -gray ... convert to grayscale
           -rgb  ... convert to red-green-blue
         -blur B ... Gaussian blur over B pixels
           -neg  ... negate pixel values
          -amp A ... multiply pixel values by A
             -lr ... flip left <-> right
             -ud ... flip up <-> down
           -prev ... delta from previous frame
'''

def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

def waitkey(ms): return chr (cv2.waitKey(ms) & 0xFF)

def parse_AxB(s): return None if s is None else tuple([int(i) for i in s.split('x')])

def parse_ROI(s): return None if s is None else tuple(map(int,re.split('[x+]',s)))

def pad(n,b): return (1+(n>>b))<<b
def nextpow2(n): return int(2**int(1+np.log2(n-1)))

def getROI(img,fig='vid'):
    cv2.imshow(fig, img)
    x,y,w,h = cv2.selectROI(fig, img, fromCenter=False, showCrosshair=True)
    ww,hh = nextpow2(w),nextpow2(h) # pad(w,5),pad(h,5) # 
    dw,dh = int((ww-w)/2),int((hh-h)/2)
    print ('ROI: %dx%d+%dx%d'%(x-dw,y-dw,ww,hh))
    return x-dw,y-dw,ww,hh

def p2heat(P): # 
    if len(P.shape) == 3: P = P.mean(axis=2)
    B = 255 * (P < 0.40) * (1-P)
    R = 255 * (P > 0.60) * P
    G = np.zeros(P.shape)
    return cv2.merge([x.astype('uint8') for x in [B,G,R]])
    
ipath,opath = getarg('-i',None),getarg('-o',None)

if not ipath or not opath:
    sys.stderr.write(usage)
    sys.exit(1)

save,show,verbose = not argp('-dry'),argp('-show'),argp('-v')
start,stop = int(getarg('-start','0')),int(getarg('-stop','0'))
size = parse_AxB(getarg('-size',None))
crop = argp('-crop')
roi = parse_ROI(getarg('-roi',None))
lr,ud = argp('-lr'),argp('-ud')
blur = int(getarg('-blur','0'))
amp = int(getarg('-amp','0'))
gray = argp('-gray')
ema = float(getarg('-ema','0'))
prev,bg = argp('-prev'),None
frame,pause,repause = 0,False,False

ofps = int(getarg('-fps','10'))
codec = getarg('-codec','XVID') # X264 H264 AVC1 XVID MJPG
fcc = cv2.VideoWriter_fourcc(*codec)

cap,out = cv2.VideoCapture(ipath),None

while True:
    key = waitkey(1)
    if key == 's':
        save = not save
        print ('frame %d -> %s' % (frame, opath if save else 'dry'))
    if key == 'i': print ('frame: %d'%frame, img.shape)
    if key == 'q': break
    if key == 'n': pause,repause = False,True
    if key == ' ': pause = not pause
    if pause: continue
    if repause: pause,repause = True,False
    ret, img = cap.read()
    if not ret: break
    frame += 1
    if frame < start: continue
    if frame > stop and stop > 0: break
    if frame == start and show: pause = True
    r,c,ch = img.shape
    if crop and not roi: roi = getROI(img)
    if roi is not None:
        x,y,w,h = roi
        img = img[y:y+h,x:x+w]
    if gray: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if lr:   img = cv2.flip(img, 1)
    if ud:   img = cv2.flip(img, 0)
    if blur: img = cv2.GaussianBlur(img,(blur,blur),0)
    if size: img = cv2.resize(img, size)
    if ema:
        add = img.astype('float32')
        if bg is None: bg = add
        img = (amp*(add-bg)*(add>bg)).astype('uint8')
        #img = (amp*(add/bg-1)*(add>bg+10)).astype('uint8')
        #img = (amp*np.abs(add-bg)).astype('uint8')
        #P = (1/(1+np.exp(bg-add)))
        #img = p2heat(P)
        bg = ema * bg + (1 - ema) * add
    if prev:
        tmp = copy.copy(img)
        img = np.abs(img-bg)
        bg = tmp
    if amp and not ema: img = img * amp
    if show: cv2.imshow('vid', img)
    if save:
        if not out: out = cv2.VideoWriter(opath, fcc, ofps, img.shape[:2], isColor=(not gray))
        out.write(img)

cap.release()
if out: out.release()
cv2.destroyAllWindows()
