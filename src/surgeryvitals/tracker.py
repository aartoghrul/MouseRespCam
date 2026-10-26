"""Keeps the region of interest fixed on the mouse by registering the rigid scene around it
(stereotax, platform, tape) to a reference frame. The region itself is masked out so the tracker
never follows - and cancels - the breathing motion.

ORB features (Rublee et al. 2011), Lowe ratio test (Lowe 2004), similarity transform fitted with
RANSAC (Fischler & Bolles 1981). See docs/ALGORITHM.md, section 4."""
import cv2
import numpy as np


class AnchorTracker:
    def __init__(self, p):
        self.p = p
        self.orb = cv2.ORB_create(1500, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_kp = self.ref_des = None
        self.box = None
        self.A = np.eye(2, 3)
        self.inliers = 0
        self.fail_frames = 0

    @property
    def initialised(self):
        return self.ref_des is not None

    def init(self, gray, box):
        s = self.p.reg_scale
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = np.full(g.shape, 255, np.uint8)
        x0, y0, x1, y1 = box
        mask[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = 0
        kp, des = self.orb.detectAndCompute(g, mask)
        self.ref_kp = np.float32([k.pt for k in kp]) / s
        self.ref_des = des
        self.box = box
        self.A = np.eye(2, 3)
        self.inliers = len(kp)
        self.fail_frames = 0

    def rekey(self, gray):
        """Refresh the reference with the current frame. A reference taken while hands filled the view
        matches poorly once they move; new keypoints are mapped back into the original reference
        coordinates (inverse of the current transform) so the region does not drift."""
        s = self.p.reg_scale
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        q = self.quad() * s
        x0, y0 = np.floor(q.min(0)).astype(int)
        x1, y1 = np.ceil(q.max(0)).astype(int)
        mask = np.full(g.shape, 255, np.uint8)
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
        kp, des = self.orb.detectAndCompute(g, mask)
        if des is None or len(kp) < 2 * self.p.min_inliers:
            return False
        Ainv = cv2.invertAffineTransform(self.A)
        self.ref_kp = cv2.transform((np.float32([k.pt for k in kp]) / s)[None], Ainv)[0]
        self.ref_des = des
        return True

    def register(self, gray):
        """Estimate the reference->current transform. Returns True if it was accepted."""
        s = self.p.reg_scale
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, des = self.orb.detectAndCompute(g, None)
        ok = False
        self.inliers = 0
        if des is not None and self.ref_des is not None and len(kp) > 20:
            pairs = [m for m in self.bf.knnMatch(self.ref_des, des, k=2)
                     if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(pairs) > 15:
                p0 = self.ref_kp[[m[0].queryIdx for m in pairs]]
                p1 = np.float32([kp[m[0].trainIdx].pt for m in pairs]) / s
                M, inl = cv2.estimateAffinePartial2D(p0, p1, ransacReprojThreshold=4)
                if M is not None:
                    self.inliers = int(inl.sum())
                    scale = np.hypot(M[0, 0], M[1, 0])
                    if self.inliers >= self.p.min_inliers and 0.8 < scale < 1.25:
                        self.A = M
                        ok = True
        self.fail_frames = 0 if ok else self.fail_frames + self.p.reg_every
        return ok

    def quad(self):
        """Current ROI corners (4x2, full-res), clockwise from top-left."""
        x0, y0, x1, y1 = self.box
        pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return cv2.transform(pts[None], self.A)[0]
