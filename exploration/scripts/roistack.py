# Dump the 4x-downsampled grayscale ROI for every frame (optionally warped by saved anchor-tracking transforms).
import cv2, numpy as np, sys
path,out=sys.argv[1],sys.argv[2]; x0,y0,x1,y1=[int(v) for v in sys.argv[3].split(',')]
T=np.load(sys.argv[4]) if len(sys.argv)>4 else None
c=cv2.VideoCapture(path); w,h=(x1-x0)//4,(y1-y0)//4
dst=np.float32([[0,0],[w,0],[0,h]]); src0=np.float32([[x0,y0],[x1,y0],[x0,y1]])
X=[]; i=0
while True:
    ok,f=c.read()
    if not ok: break
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    if T is None: r=cv2.resize(g[y0:y1,x0:x1],(w,h),interpolation=cv2.INTER_AREA)
    else:
        A=T[max(i-1,0)].reshape(2,3); src=cv2.transform(src0[None],A)[0]
        r=cv2.warpAffine(g,cv2.getAffineTransform(src,dst),(w,h),flags=cv2.INTER_AREA)
    X.append(r); i+=1
np.save(out,np.array(X)); print(out,np.array(X).shape)
