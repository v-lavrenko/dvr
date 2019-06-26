#!/usr/local/bin/python

import numpy as np
import cv2, sys
from datetime import datetime as dt
from datetime import timedelta as td
from time import sleep

def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

def die():
    vid.release()
    cap.release()
    cv2.destroyAllWindows()
    print 'result:', opath

def grid(caps):
    return None

def waitkey(ms): return chr (cv2.waitKey(ms) & 0xFF)

import signal
signal.signal(signal.SIGINT, die)
signal.signal(signal.SIGTERM, die)

if len(sys.argv) < 2:
    sys.stderr.write('OpenCV'+cv2.__version__+' [Q] to quit\n')
    sys.stderr.write('Usage: dvr.py [-hide] [-gray] [-flip] [-4x] [-2x] [-wait 600] [-codec XVID] [-o out.avi]\n')

plot  = argp('-plot')
stamp = getarg('-stamp','full')
head  = getarg('-head','')
cams  = int(getarg('-cams','0'))
icam  = int(getarg('-cam','0'))
inpt  = getarg('-in','')
show  = not argp('-hide')
save  = not argp('-dry')
lr  = argp('-lr')
ud  = argp('-ud')
gray  = argp('-gray')
blur  = int(getarg('-blur','1'))
amp   = int(getarg('-amp','0'))
wait  = int(getarg('-wait','100000'))
stop  = dt.now() + td(seconds=wait)
opath = getarg('-o',dt.now().strftime("dvr_%F_%T.avi"))
codec = getarg('-codec','XVID') # X264 H264 AVC1 XVID MJPG
osize = (1280,720) if argp('-4x') else (640,360) if argp('-2x') else (320,180)
fps = int(getarg('-fps','20')) # control playback speed
zoom,ROI,CLR = None,{},{'a':(255,0,255), 'b':(255,255,0), 'c':(0,255,255), 'd':(255,255,255)}
fcc = cv2.VideoWriter_fourcc(*codec)
vid = cv2.VideoWriter(opath, fcc, fps, osize, isColor=(not gray))

motion,noise,gotnoise = False,False,False
fgbg = cv2.createBackgroundSubtractorMOG2 (history=500,
                                           varThreshold=16,
                                           detectShadows=False)

cap = cv2.VideoCapture(inpt if inpt else icam)
#cap.set(3, 1280) # set the resolution
#cap.set(4, 720)
#cap.set(cv2.CAP_PROP_AUTOFOCUS,0) # turn the autofocus off
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0)

scale,thick,white,black,green,blue = 1,1,(255,255,255),(0,0,0),(0,255,0),(255,0,0)
font = cv2.FONT_HERSHEY_PLAIN
frame = 0
if plot:
    canvas,xxx = np.zeros((128, osize[0], 3), dtype="uint8"),0

#sleep(5)

while(dt.now() < stop):
    ret, img = cap.read()  # capture frame-by-frame
    if not ret: break
    if gray: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if lr: img = cv2.flip(img, 1)
    if ud: img = cv2.flip(img, 0)
    img = cv2.resize(img, osize)
    if   stamp == 'full': text = dt.now().strftime("%F %T.%f")[:22] + ' ' + head
    elif stamp == 'frame': text = 'frame: ' + str(frame)
    elif stamp == 'none': text = ''
    if zoom:
        x,y,w,h = zoom
        crop = img[y:y+h,x:x+w]
        img = cv2.resize(crop, osize)
    if blur > 1:
        img = cv2.GaussianBlur(img,(blur,blur),0)
    if gotnoise:
        fgmask = fgbg.apply(img,learningRate=(0.01 if noise else 0))
        #img = fgmask
        img [fgmask < 1] = 0
    if motion:
        fgmask = fgbg.apply(img,learningRate=0.01)
        img [fgmask < 1] = 0
    if amp > 1:
        img = img * amp
    if plot:
        xxx = (xxx + 1) % osize[0]
        canvas[:,xxx,:] = 0
        #if xxx == 0: canvas[:] = 0
    for key in ROI:
        x,y,w,h = ROI[key]
        crop = img[y:y+h,x:x+w]
        cv2.rectangle(img, (x,y), (x+w,y+h), CLR[key], thickness=1)
        lumi = int(10. * crop.sum() / crop.size)
        text = text + (' %s:%d' % (key, lumi))
        if plot:
            canvas[128-lumi/20,xxx] = CLR[key]
            img[0:128] = canvas
        #scale = 3
        #cv2.circle(img, (xxx,yyy), 1, CLR[key])
    if not save: text += ' dry'
    if noise: text += ' noise'
    if amp > 1: text += ' x'+str(amp)
    if blur > 1: text += ' ~'+str(blur)
    #cv2.rectangle(img, (0,0), (105,20), black, thickness=cv2.FILLED)
    cv2.putText(img, text, (5,5+10*scale), font, scale, black, 3)
    cv2.putText(img, text, (5,5+10*scale), font, scale, white, 1)
    
    if show:
        title = opath # if save else 'dry'
        cv2.imshow(title,img)  # display the resulting frame

    if save:
        vid.write(img)
        frame += 1

    key = chr (cv2.waitKey(1) & 0xFF)
    if key == 'q':
        break
    if key in 'abcd':
        ROI[key] = cv2.selectROI(title, img, fromCenter=False, showCrosshair=True)
        if ROI[key] == (0,0,0,0): del ROI[key]
    if key == 'z':
        zoom = cv2.selectROI(title, img, fromCenter=False, showCrosshair=True)
        if zoom == (0,0,0,0): zoom = None
    if key == '0' and plot:
        xxx = 0
        canvas[:] = 0
    if key == 's': save = not save
    if key == 'n':
        noise = not noise
        gotnoise = True
    if key == 'm': motion = not motion
    if key == '+': amp += 1
    if key == '-': amp -= 1
    if key == '<': blur -= 2
    if key == '>': blur += 2
    if key == 'l':
        head = raw_input('Label: ')
        #head = sys.stdin.readline()[:-1]

die()
