# Anchor-based ROI tracking: register the rigid scene around the ROI (ORB+RANSAC similarity)
# to a reference frame, map ROI through the transform, then compute motion features in the warped ROI.
import cv2, numpy as np, sys, time
path,out=sys.argv[1],sys.argv[2]; ref_fr=int(sys.argv[3]); x0,y0,x1,y1=[int(v) for v in sys.argv[4].split(',')]
c=cv2.VideoCapture(path); fps=c.get(5); N=int(c.get(7))
c.set(1,ref_fr); ok,ref=c.read(); c.set(1,0)
S=0.5  # registration scale
orb=cv2.ORB_create(1500,fastThreshold=10); bf=cv2.BFMatcher(cv2.NORM_HAMMING)
rg=cv2.cvtColor(cv2.resize(ref,None,fx=S,fy=S),cv2.COLOR_BGR2GRAY)
mask=np.full(rg.shape,255,np.uint8); mask[int(y0*S):int(y1*S),int(x0*S):int(x1*S)]=0  # ignore the breathing ROI itself
kr,dr=orb.detectAndCompute(rg,mask)
A=np.eye(2,3); inl=0; prev=None; F=[]; T=[]; t0=time.time()
w,h=(x1-x0)//4,(y1-y0)//4
dst=np.float32([[0,0],[w,0],[0,h]]); src0=np.float32([[x0,y0],[x1,y0],[x0,y1]])
for i in range(N):
    ok,f=c.read()
    if not ok: break
    if i%15==0:  # re-register every 0.5 s
        g=cv2.cvtColor(cv2.resize(f,None,fx=S,fy=S),cv2.COLOR_BGR2GRAY)
        k,d=orb.detectAndCompute(g,None)
        if d is not None and len(k)>20:
            m=[a for a,b in bf.knnMatch(dr,d,k=2) if a.distance<0.8*b.distance]
            if len(m)>15:
                p0=np.float32([kr[a.queryIdx].pt for a in m])/S; p1=np.float32([k[a.trainIdx].pt for a in m])/S
                M,inm=cv2.estimateAffinePartial2D(p0,p1,ransacReprojThreshold=4)
                inl=int(inm.sum()) if M is not None else 0
                if M is not None and inl>=25: A=M   # else: hold last good transform (occlusion)
            else: inl=len(m)
    src=cv2.transform(src0[None],A)[0]
    Wm=cv2.getAffineTransform(src,dst)
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    r=cv2.warpAffine(g,Wm,(w,h),flags=cv2.INTER_AREA)
    if prev is not None:
        fl=cv2.calcOpticalFlowFarneback(prev,r,None,0.5,2,9,3,5,1.1,0)
        F.append([fl[...,1].mean(),fl[...,0].mean(),np.median(fl[...,1]),np.abs(r.astype(np.float32)-prev).mean(),r.mean(),inl])
        T.append(A.ravel())
    prev=r
    if i in (300,4860,9720,12000,14500):
        cv2.imwrite(f'exploration/trk_{i}.jpg',cv2.resize(r,None,fx=3,fy=3))
F=np.array(F); np.save(out,F); np.save(out.replace('.npy','_T.npy'),np.array(T)); print(F.shape,f'{time.time()-t0:.0f}s')
