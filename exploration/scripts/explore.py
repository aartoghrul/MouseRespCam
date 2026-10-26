import cv2, numpy as np, sys
from scipy import signal
path=sys.argv[1]; x0,y0,x1,y1=1120,800,1400,980   # mouse back
bx0,by0,bx1,by1=1450,820,1560,1000                # stereotax post (rigid reference)
c=cv2.VideoCapture(path); fps=c.get(5)
prev=None; prevb=None
mean=[];diff=[];fy=[];fx=[];bfy=[];bfx=[];pix=[]
while True:
    ok,f=c.read()
    if not ok: break
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    r=cv2.resize(g[y0:y1,x0:x1],None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA).astype(np.float32)
    b=cv2.resize(g[by0:by1,bx0:bx1],None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA).astype(np.float32)
    mean.append(r.mean()); pix.append(r.ravel())
    if prev is not None:
        diff.append(np.abs(r-prev).mean())
        fl=cv2.calcOpticalFlowFarneback(prev.astype(np.uint8),r.astype(np.uint8),None,0.5,2,9,3,5,1.1,0)
        fy.append(fl[...,1].mean()); fx.append(fl[...,0].mean())
        flb=cv2.calcOpticalFlowFarneback(prevb.astype(np.uint8),b.astype(np.uint8),None,0.5,2,9,3,5,1.1,0)
        bfy.append(flb[...,1].mean()); bfx.append(flb[...,0].mean())
    prev=r; prevb=b
pix=np.array(pix)
# PCA of detrended pixel series
P=signal.detrend(pix,axis=0); P=signal.sosfiltfilt(signal.butter(2,[0.2,5],'band',fs=fps,output='sos'),P,axis=0)
U,S,Vt=np.linalg.svd(P,full_matrices=False)
sigs={'mean':np.array(mean),'diff':np.array(diff),'flow_y_cum':np.cumsum(fy),'flow_x_cum':np.cumsum(fx),
      'ref_flow_y_cum':np.cumsum(bfy),'ref_flow_x_cum':np.cumsum(bfx),'pca1':U[:,0]*S[0],'pca2':U[:,1]*S[1]}
print('fps',fps,'frames',len(mean),'pca var expl',(S[:4]**2/np.sum(S**2)).round(3))
sos=signal.butter(2,[0.3,4],'band',fs=fps,output='sos')
np.savez('exploration/sigs.npz',fps=fps,**sigs)
for k,v in sigs.items():
    v=signal.detrend(v); vf=signal.sosfiltfilt(sos,v)
    f,p=signal.welch(vf,fs=fps,nperseg=min(len(vf),256),nfft=2048)
    m=(f>0.3)&(f<4); pk=f[m][np.argmax(p[m])]
    # peak count
    pks,_=signal.find_peaks(vf,distance=fps*0.25,prominence=np.std(vf)*0.5)
    print(f'{k:16s} peakHz={pk:.2f} ({pk*60:.0f} bpm)  peaks/min={len(pks)/(len(vf)/fps)*60:.0f}  std={np.std(v):.4f}')
