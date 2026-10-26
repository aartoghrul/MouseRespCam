import cv2, numpy as np, glob
tiles=[]
for f in sorted(glob.glob('videos/Recording_*.mp4')):
    c=cv2.VideoCapture(f); n=int(c.get(7)); row=[]
    for fr in np.linspace(0,n-30,4).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES,fr); ok,im=c.read()
        im=cv2.resize(im,(480,270)); cv2.putText(im,f'{f[-5]} t={fr/30:.0f}s',(5,20),0,0.6,(0,0,255),2); row.append(im)
    tiles.append(np.hstack(row))
cv2.imwrite('exploration/montage.jpg',np.vstack(tiles))
