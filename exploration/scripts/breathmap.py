# Auto-localise the breathing region: whole-frame dense flow at 1/4 res, per-block periodicity in the breathing band.
import cv2, numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import signal
path,t0,dur,out=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
c=cv2.VideoCapture(path); fps=c.get(5); c.set(1,int(t0*fps)); N=int(dur*fps); B=8  # block = 8x8 px at 1/4 res (32 px full res)
prev=None; S=[]
for i in range(N):
    ok,f=c.read(); g=cv2.cvtColor(cv2.resize(f,(480,272),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    if i==N//2: mid=cv2.cvtColor(cv2.resize(f,(480,272)),cv2.COLOR_BGR2RGB)
    if prev is not None:
        fl=cv2.calcOpticalFlowFarneback(prev,g,None,0.5,3,9,3,5,1.1,0)
        S.append(fl.reshape(272//B,B,480//B,B,2).mean((1,3)))
    prev=g
S=np.cumsum(np.array(S),axis=0)                       # (T, by, bx, 2) displacement per block
S=signal.sosfiltfilt(signal.butter(2,[0.1,5],'band',fs=fps,output='sos'),S,axis=0)
T,by,bx,_=S.shape; f=np.fft.rfftfreq(4096,1/fps); m=(f>0.5)&(f<3.5)  # 30-210 bpm: excludes human (surgeon) breathing ~12-20 bpm
score=np.zeros((by,bx)); rate=np.zeros((by,bx)); amp=np.zeros((by,bx))
for y in range(by):
    for x in range(bx):
        v=S[:,y,x]; u,s_,vt=np.linalg.svd(v-v.mean(0),full_matrices=False); v=u[:,0]*s_[0]   # dominant motion direction
        P=np.abs(np.fft.rfft(v*np.hanning(T),4096))**2; Hs=P.copy()
        for k in (2,3): Hs[:len(P)//k]+=P[::k][:len(P)//k]
        fh=f[m][np.argmax(Hs[m])]
        score[y,x]=sum(P[(f>k*fh-.08)&(f<k*fh+.08)].sum() for k in (1,2,3))/P[(f>.1)&(f<6)].sum()  # slow (surgeon) power counts against the block
        rate[y,x]=fh*60; amp[y,x]=np.std(v)
# candidate regions: connected blocks that are periodic, above the noise floor, and share one rate (±4 bpm)
act=score*(amp>3*np.percentile(amp,10))   # above the static-background noise floor (not relative to the busy surgeon)
thr=min(0.5,max(0.3,np.percentile(act,98)*0.8))   # adaptive: weaker signals still qualify if they stand out
best=None
for r0 in np.unique(np.round(rate[act>0.5])):
    cand=((act>thr)&(np.abs(rate-r0)<4)).astype(np.uint8)
    n_,lab=cv2.connectedComponents(cand,connectivity=8)
    for k in range(1,n_):
        reg=lab==k; val=act[reg].mean()*np.sqrt(reg.sum())   # strong + coherent, but area only counts as sqrt
        if reg.sum()>=3 and (best is None or val>best[0]): best=(val,reg,np.median(rate[reg]))
_,mask,dom=best
ys,xs=np.nonzero(mask); box=(xs.min()*B,ys.min()*B,(xs.max()+1)*B,(ys.max()+1)*B)
fig,ax=plt.subplots(1,3,figsize=(18,4.2))
ax[0].imshow(mid); ax[0].imshow(cv2.resize(act,(480,272),interpolation=cv2.INTER_NEAREST),alpha=.55,cmap='magma',vmin=0,vmax=1); ax[0].set_title('periodicity score (breathing band)')
ax[1].imshow(mid); ax[1].imshow(np.ma.masked_where(~cv2.resize(mask.astype(np.uint8),(480,272),interpolation=cv2.INTER_NEAREST).astype(bool),np.ones((272,480))),alpha=.6,cmap='autumn'); ax[1].add_patch(plt.Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,ec='lime',lw=2)); ax[1].set_title(f'auto-selected breathing region @ {dom:.0f} bpm')
ax[2].imshow(np.where(act>0.5,rate,np.nan),cmap='viridis',vmin=0,vmax=150); ax[2].set_title('dominant rate per block (bpm, score>0.5)')
for a in ax: a.axis('off')
fig.suptitle(f'{path.split("/")[-1]}  t={t0:.0f}–{t0+dur:.0f}s'); plt.tight_layout(); plt.savefig(out,dpi=65)
print(out,f'region {dom:.0f} bpm, {mask.sum()} blocks, box(full-res)={[v*4 for v in box]}')
