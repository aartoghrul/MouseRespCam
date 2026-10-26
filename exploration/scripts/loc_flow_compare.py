# Does a cheaper flow (DIS) find the same breathing region as Farneback? Result: DIS misses weak breathing (Rec1 t=200); block-LK (removed) missed all of Rec1.
import sys, numpy as np
from surgeryvitals.config import Params
from surgeryvitals.localizer import BreathLocalizer
from surgeryvitals.source import VideoFileSource
cases=[tuple(map(int,c.split(':'))) for c in sys.argv[1:]] or [(1,60),(1,120),(1,200),(2,300),(3,400),(4,392),(5,200),(5,380)]
for rec,t0 in cases:
    out=[]
    for flow in ('farneback','dis'):
        p=Params(loc_flow=flow); src=VideoFileSource(f'videos/Recording_{rec}.mp4'); src.cap.set(1,int(t0*src.fps))
        loc=BreathLocalizer(src.shape,src.fps,p)
        for i in range(int(20*src.fps)):
            f,_=src.read()
            if i%p.loc_stride==0: loc.push(f[...,1])
        pr=loc.localize(); out.append(pr)
    a,b=out
    fmt=lambda q: 'none' if q is None else f'{q.bpm:5.1f} bpm score {q.score:.2f} box {q.box}'
    print(f'Rec{rec} t={t0:3d}  farneback: {fmt(a)}\n            dis      : {fmt(b)}')
