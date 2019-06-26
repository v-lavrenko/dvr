#!/usr/local/bin/python

import numpy as np
import cv2, sys, os
#from matplotlib import pyplot as plt

def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

_ref,_hyp = getarg('-ref','ref.png'),getarg('-hyp','hyp.png')
ref,hyp = cv2.imread(_ref,False),cv2.imread(_hyp,False)
out = getarg('-out',None)
smooth = float(getarg('-smooth','0.01'))

#print _ref+':', ref.shape, _hyp+':', hyp.shape

#if len(hyp.shape) == 3: hyp = hyp.max(axis=2)
if hyp.shape != ref.shape: ref = cv2.resize(ref, hyp.shape[::-1])

if out:
    diff = ((255. + hyp - ref) / 2).astype('uint8')
    if out:  cv2.imwrite(out,diff)

pos,neg = hyp[ref>127],hyp[ref<128]
Ep,En = pos.mean(), neg.mean()
Lp,Ln = np.log(pos/255.+smooth).sum(), np.log(1-neg/255.+smooth).sum()
entr = (-Lp -Ln) / (len(pos) + len(neg))
perp = - Lp / len(pos) - Ln / len(neg)

print '%.2f\t%.4f\t%.0f\t%.0f\t%.0f\t%s' % (perp, entr, Ep-En, Ep, En, getarg('-hyp',''))
