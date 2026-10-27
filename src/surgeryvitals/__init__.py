"""Webcam respiratory-rate monitoring for anaesthetised mice."""
import os

# Windows Media Foundation: without this, some cameras take 10-30 s to open (read before cv2 loads)
os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')
