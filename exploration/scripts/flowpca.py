# Facemap idea applied in flow space: PCA of the dense optical-flow field (signed, brightness-invariant),
# basis learned on a calibration window. Also reports a reference-free periodicity score for each signal.
import numpy as np, sys, cv2
from scipy import signal
sys.argv_saved=sys.argv; stack,feat,cal0,cal1=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
fps=30.; X=np.load(stack); n=len(X); c0,c1=int(cal0*fps),int(cal1*fps)
Fl=np.zeros((n,X.shape[1]*X.shape[2]*2),np.float32)
for i in range(1,n): Fl[i]=cv2.calcOpticalFlowFarneback(X[i-1],X[i],None,0.5,2,9,3,5,1.1,0).ravel()
U,S,Vt=np.linalg.svd(Fl[c0:c1],full_matrices=False); T=np.cumsum(Fl@Vt[:10].T,axis=0)
sos=signal.butter(2,[0.25,5],'band',fs=fps,output='sos')
def score(x,W=int(15*fps)):
    f=np.fft.rfftfreq(8192,1/fps); m=(f>.25)&(f<4); r=[];q=[]
    for s in range(0,len(x)-W,int(fps)):
        v=x[s:s+W]; P=np.abs(np.fft.rfft((v-v.mean())*np.hanning(W),8192))**2; Hs=P.copy()
        for k in (2,3): Hs[:len(P)//k]+=P[::k][:len(P)//k]
        fh=f[m][np.argmax(Hs[m])]; r.append(fh*60); q.append(sum(P[(f>k*fh-.1)&(f<k*fh+.1)].sum() for k in (1,2,3))/P[(f>.25)&(f<6)].sum())
    return np.array(r),np.array(q)
j=max(range(10),key=lambda j: np.median(score(signal.sosfiltfilt(sos,T[c0:c1,j]))[1]))
F=np.load(feat); flow=np.r_[0,np.cumsum(F[:,0])]
rf,qf=score(signal.sosfiltfilt(sos,flow)); rp,qp=score(signal.sosfiltfilt(sos,signal.detrend(T[:,j])))
print(f'flow-PCA comp {j} (var expl {S[j]**2/np.sum(S**2):.2f}): within ±3 bpm of mean-vertical-flow: {np.mean(np.abs(rp-rf)<=3)*100:.1f}%')
print(f'median periodicity score  mean-vertical-flow {np.median(qf):.3f}   flow-PCA {np.median(qp):.3f}')
np.save(stack.replace('_stack','_flowpca'),T[:,j])
