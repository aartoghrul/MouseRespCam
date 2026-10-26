# Facemap-style SVD variants vs optical flow. Basis learned on a calibration window only (as a live system would),
# then every frame is projected onto it. Rate per 15 s window via harmonic-sum spectrum.
import numpy as np, sys, time, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import signal
stack,feat,cal0,cal1,out,title=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),sys.argv[5],sys.argv[6]
fps=30.; X=np.load(stack).astype(np.float32); n,h,w=X.shape; X=X.reshape(n,-1)
F=np.load(feat); flow=np.r_[0,np.cumsum(F[:,0])]
c0,c1=int(cal0*fps),int(cal1*fps); K=10
sos=signal.butter(2,[0.25,5],'band',fs=fps,output='sos')
def hs_rate(x,W=int(15*fps),H=int(fps)):
    t,r,q=[],[],[]
    f=np.fft.rfftfreq(8192,1/fps); m=(f>0.25)&(f<4)
    for s in range(0,len(x)-W,H):
        v=x[s:s+W]; P=np.abs(np.fft.rfft((v-v.mean())*np.hanning(W),8192))**2; Hs=P.copy()
        for k in (2,3): Hs[:len(P)//k]+=P[::k][:len(P)//k]
        fh=f[m][np.argmax(Hs[m])]; t.append((s+W/2)/fps); r.append(fh*60)
        q.append(sum(P[(f>k*fh-.1)&(f<k*fh+.1)].sum() for k in (1,2,3))/P[(f>.25)&(f<6)].sum())
    return np.array(t),np.array(r),np.array(q)
def pick(T):  # choose the component with the most periodic (harmonic-concentrated) calibration segment
    best=max(range(T.shape[1]),key=lambda j: np.median(hs_rate(signal.sosfiltfilt(sos,T[c0:c1,j]))[2]))
    return best
variants={}
tm={}
# 1) Facemap motSVD: |frame diff|, mean-subtracted, SVD
D=np.abs(np.diff(X,axis=0)); D=np.vstack([D[:1],D])
mu=D[c0:c1].mean(0); U,S,Vt=np.linalg.svd(D[c0:c1]-mu,full_matrices=False); T=(D-mu)@Vt[:K].T
j=pick(T); variants[f'Facemap motSVD (|diff|), comp {j}']=T[:,j]
# 2) signed-diff SVD, integrated -> displacement-like
Ds=np.diff(X,axis=0); Ds=np.vstack([Ds[:1]*0,Ds])
U,S,Vt=np.linalg.svd(Ds[c0:c1],full_matrices=False); T=np.cumsum(Ds@Vt[:K].T,axis=0)
j=pick(T); variants[f'signed-diff SVD (integrated), comp {j}']=T[:,j]
# 3) Facemap movSVD: raw frames, mean-subtracted
mu=X[c0:c1].mean(0); U,S,Vt=np.linalg.svd(X[c0:c1]-mu,full_matrices=False); T=(X-mu)@Vt[:K].T
j=pick(T); variants[f'Facemap movSVD (raw frames), comp {j}']=T[:,j]; Vmov=Vt[:K]
variants['optical flow (vertical, integrated)']=flow
# per-frame cost: projection vs Farneback on this ROI size
a,b=X[100].reshape(h,w).astype(np.uint8),X[101].reshape(h,w).astype(np.uint8)
t0=time.perf_counter(); [ (np.abs(X[101]-X[100])-mu)@Vmov.T for _ in range(2000)]; tp=(time.perf_counter()-t0)/2000
t0=time.perf_counter(); [cv2.calcOpticalFlowFarneback(a,b,None,0.5,2,9,3,5,1.1,0) for _ in range(500)]; tf=(time.perf_counter()-t0)/500
print(f'{title}: ROI {w}x{h}; per-frame cost  SVD projection {tp*1e6:.0f} us   Farneback {tf*1e6:.0f} us')
# evaluation
tr,rf,qf=hs_rate(signal.sosfiltfilt(sos,flow)); good=qf>np.percentile(qf,30)
fig,ax=plt.subplots(len(variants),3,figsize=(18,2.6*len(variants)),gridspec_kw={'width_ratios':[1.2,2,1]})
z0=int((cal1+20)*fps); z1=z0+int(20*fps)
for i,(k,v) in enumerate(variants.items()):
    x=signal.sosfiltfilt(sos,signal.detrend(v)); t,r,q=hs_rate(x)
    agree=np.mean(np.abs(r[good]-rf[good])<=3)*100; dbl=np.mean(np.abs(r[good]-2*rf[good])<=4)*100; half=np.mean(np.abs(r[good]-rf[good]/2)<=3)*100
    print(f'  {k:42s} within ±3 bpm of flow: {agree:5.1f}%   at 2x: {dbl:5.1f}%   at 1/2x: {half:5.1f}%')
    tt=np.arange(z0,z1)/fps; ax[i,0].plot(tt,x[z0:z1],lw=.8); ax[i,0].set_title(k,fontsize=9,loc='left')
    ax[i,1].plot(np.arange(len(x))/fps,x,lw=.3); ax[i,1].axvspan(cal0,cal1,color='g',alpha=.15); ax[i,1].set_ylim(np.percentile(x,[0.5,99.5])*2)
    ax[i,2].plot(tr,rf,'k.',ms=2,label='flow ref'); ax[i,2].plot(t,r,'.',ms=3,label=f'this ({agree:.0f}% agree)'); ax[i,2].set_ylim(0,200); ax[i,2].legend(fontsize=7)
ax[0,1].set_title('full recording (green = calibration window used to learn SVD basis)',fontsize=9)
ax[0,2].set_title('rate, bpm (15 s windows)',fontsize=9); ax[-1,0].set_xlabel('s (20 s excerpt)'); ax[-1,1].set_xlabel('s'); ax[-1,2].set_xlabel('s')
fig.suptitle(title); plt.tight_layout(); plt.savefig(out,dpi=60)
