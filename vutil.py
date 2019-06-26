
def argp(key): return key in sys.argv

def getarg(key,default):
    for i in range(len(sys.argv)):
        if sys.argv[i] == key: return sys.argv[i+1]
    return default

def waitkey(ms): return chr (cv2.waitKey(ms) & 0xFF)

def visualize(imgs,wait=1000,tag='haha'):
    font = cv2.FONT_HERSHEY_PLAIN
    imgs = [cv2.resize(x.astype('uint8'), (320,180)) for x in imgs]
    img = np.concatenate(imgs,axis=1)
    cv2.putText(img, tag, (5,15), font, 1, (0,0,0), 3)
    cv2.putText(img, tag, (5,15), font, 1, (255,255,255), 1)
    cv2.imshow('xxx', img)
    waitkey(wait)

def nextp2(n): return 1<<int(1+np.log2(n-1))

def pad(n,b): return (1+(n>>b))<<b
