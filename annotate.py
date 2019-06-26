#!/usr/local/bin/python

import numpy as np
import cv2, sys, os, copy

def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

def die():
    if save: ovid.release()
    cap.release()
    cv2.destroyAllWindows()
    print 'result:', opath

def waitkey(ms): return chr (cv2.waitKey(ms) & 0xFF)

def cap_size(cap): return (int(cap.get(3)), int(cap.get(4)))

import signal
signal.signal(signal.SIGINT, die)
signal.signal(signal.SIGTERM, die)

color = {'r':(0,0,255), 'b':(255,0,0), 'g':(0,255,0), 'w':(255,255,255)}
amp = int(getarg('-amp','1'))
pause,repause = False,True
ipath,opath = getarg('-in',None),getarg('-out',None)
save = argp('-save')
codec = getarg('-codec','XVID') # X264 H264 AVC1 XVID MJPG
if not ipath or not opath:
    sys.stderr.write('Usage: annotate.py -in x.avi -out y.avi\n')
    sys.exit(1)

cap,img,omg = cv2.VideoCapture(ipath),None,None
osize,frame = cap_size(cap),1

if save: # open output files, unless dry-run
    fcc = cv2.VideoWriter_fourcc(*codec)
    ovid = cv2.VideoWriter(opath, fcc, 20, osize, isColor=True)

print '''
space ... pause/continue
    n ... next frame
    q ... quit
 rgbw ... annotate Red/Green/Blue/White region
    x ... remove all annotations in current frame
   +- ... amplify/de-amplify frames (buggy)
'''

while True:
    key = chr (cv2.waitKey(1) & 0xFF)
    if key in 'rgbw': # annotate a region: Red,Grn,Blu,White
        x,y,w,h = cv2.selectROI(ipath, img, fromCenter=False, showCrosshair=True)
        if (x,y,w,h) == (0,0,0,0): pass # wtf?
        #cv2.rectangle(img, (x,y), (x+w,y+h), color[key], thickness=+2) # 
        #cv2.rectangle(omg, (x,y), (x+w,y+h), color[key], thickness=-1) # CV_FILLED=-1
        cv2.ellipse(img, (x+w/2, y+h/2), (w/2,h/2), 0, 0, 360, color[key], thickness=2)
        cv2.ellipse(omg, (x+w/2, y+h/2), (w/2,h/2), 0, 0, 360, color[key], thickness=-1)
    if key == 'x': # cancel annotations in current frame
        img = copy.copy(undo)
        omg = np.zeros(img.shape).astype('uint8')
    if key in '+-':
        damp = +1 if key == '+' else -1
        amp += damp
        if amp < 1: amp = 1
        img = (img / (amp - damp)) * amp
    if key == 'n': pause,repause = False,True # next frame
    if key == 'q': break # quit
    if key == ' ': pause = not pause # pause
    if img is not None: cv2.imshow(ipath,img)
    #if omg is not None: cv2.imshow(opath,omg)
    if pause: continue
    if repause: pause,repause = True,False
    
    if save and omg is not None: ovid.write(omg)
    
    ret, img = cap.read()
    if not ret: break
    frame += 1

    if amp > 1: img = img * amp
    undo = copy.copy(img) # keep original for undo
    omg = np.zeros(img.shape).astype('uint8')

die()
