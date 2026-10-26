import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import signal
F=np.load(sys.argv[1]); fps=30.; out=sys.argv[2]
fy=F[:,0]; diff=F[:,3]; ncc=F[:,5]; br=F[:,4]
# displacement = cumulative flow, high-passed
sos=signal.butter(2,[0.25,5],'band',fs=fps,output='sos')
x=signal.sosfiltfilt(sos,np.cumsum(fy))
W=int(15*fps); H=int(fps); lo,hi=0.25,4.0
t=[];spec=[];acf=[];pk=[];sqi=[];S=[]
for s in range(0,len(x)-W,H):
    w=x[s:s+W]*np.hanning(W)
    f=np.fft.rfftfreq(8192,1/fps); P=np.abs(np.fft.rfft(w-w.mean(),8192))**2
    m=(f>lo)&(f<hi); fp=f[m][np.argmax(P[m])]
    # harmonic-sum spectrum (pitch-detection style): favours fundamental
    Hs=P.copy()
    for k in (2,3): Hs[:len(P)//k]+=P[::k][:len(P)//k]
    fh=f[m][np.argmax(Hs[m])]
    # autocorrelation
    v=x[s:s+W]-x[s:s+W].mean(); a=np.correlate(v,v,'full')[W-1:]; a/=a[0]
    lags=np.arange(W)/fps; mm=(lags>1/hi)&(lags<1/lo); la=lags[mm][np.argmax(a[mm])]
    # peak detection on negative deflections (inspiration dips)
    p,_=signal.find_peaks(-v,distance=int(fps/hi),prominence=np.std(v)*0.8)
    pr=60*fps/np.median(np.diff(p)) if len(p)>2 else np.nan
    # spectral SQI: power fraction within ±0.1Hz of fundamental+harmonics
    band=sum(P[(f>k*fh-0.1)&(f<k*fh+0.1)].sum() for k in (1,2,3)); q=band/P[(f>lo)&(f<6)].sum()
    t.append((s+W/2)/fps); spec.append(fp*60); acf.append(60/la); pk.append(pr); sqi.append(q); S.append(Hs[m]/Hs[m].max())
t=np.array(t)
fig,ax=plt.subplots(5,1,figsize=(16,13),sharex=True)
tt=np.arange(len(x))/fps
ax[0].plot(tt,x,lw=.4); ax[0].set_ylabel('displacement (flow_y)'); ax[0].set_ylim(np.percentile(x,[0.5,99.5])*1.5)
ax[1].plot(tt,diff,lw=.4,label='ROI |diff|'); ax2=ax[1].twinx(); ax2.plot(tt,ncc,'r',lw=.6); ax2.set_ylabel('NCC vs first frame',color='r'); ax[1].set_ylabel('ROI diff energy')
ax[2].imshow(np.array(S).T,aspect='auto',origin='lower',extent=[t[0],t[-1],lo*60,hi*60],cmap='magma'); ax[2].set_ylabel('harmonic-sum spectrogram (bpm)')
ax[3].plot(t,spec,'.',ms=3,label='Welch peak'); ax[3].plot(t,acf,'.',ms=3,label='autocorr'); ax[3].plot(t,pk,'.',ms=3,label='peak count'); ax[3].legend(); ax[3].set_ylabel('bpm'); ax[3].set_ylim(0,200)
ax[4].plot(t,sqi); ax[4].set_ylabel('spectral SQI'); ax[4].set_xlabel('s')
plt.tight_layout(); plt.savefig(out,dpi=60)
print('median bpm  spec %.1f acf %.1f peaks %.1f'%(np.nanmedian(spec),np.nanmedian(acf),np.nanmedian(pk)))
