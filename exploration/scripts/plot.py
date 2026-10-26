import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import signal
d=np.load('exploration/sigs.npz'); fps=float(d['fps'])
keys=[k for k in d.files if k!='fps']
fig,ax=plt.subplots(len(keys),2,figsize=(14,2*len(keys)))
sos=signal.butter(2,[0.3,4],'band',fs=fps,output='sos')
for i,k in enumerate(keys):
    v=signal.detrend(d[k]); t=np.arange(len(v))/fps
    ax[i,0].plot(t,v,lw=.7,color='0.6'); ax[i,0].plot(t,signal.sosfiltfilt(sos,v),lw=1); ax[i,0].set_ylabel(k,fontsize=8)
    f,p=signal.welch(signal.sosfiltfilt(sos,v),fs=fps,nperseg=256,nfft=2048); ax[i,1].plot(f*60,p); ax[i,1].set_xlim(0,240)
ax[-1,0].set_xlabel('s'); ax[-1,1].set_xlabel('breaths/min')
plt.tight_layout(); plt.savefig('exploration/sigs.png',dpi=70)
