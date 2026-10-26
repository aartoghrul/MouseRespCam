import cv2, numpy as np, sys, time
path,out=sys.argv[1],sys.argv[2]; x0,y0,x1,y1=[int(v) for v in sys.argv[3].split(',')]
c=cv2.VideoCapture(path); fps=c.get(5)
prev=None; ref=None; F=[]; t0=time.time()
while True:
    ok,f=c.read()
    if not ok: break
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    r=cv2.resize(g[y0:y1,x0:x1],None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
    if ref is None: ref=r.astype(np.float32)
    if prev is not None:
        fl=cv2.calcOpticalFlowFarneback(prev,r,None,0.5,2,9,3,5,1.1,0)
        rf=r.astype(np.float32); ncc=np.corrcoef(rf.ravel(),ref.ravel())[0,1]
        F.append([fl[...,1].mean(),fl[...,0].mean(),np.median(fl[...,1]),np.abs(rf-prev).mean(),r.mean(),ncc])
    prev=r
F=np.array(F); np.save(out,F); print(path,F.shape,f'{time.time()-t0:.0f}s')
