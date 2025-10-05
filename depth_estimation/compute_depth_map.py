"""Compute disparity and optionally convert to depth using stereo params.

Requires stereo params saved by `calibration/stereo_calibration.py` or calibration tool.
"""
import cv2
import numpy as np
import argparse

def compute_disparity(left, right, use_sgbm=True):
    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    if use_sgbm:
        window_size = 5
        min_disp = 0
        num_disp = 16*10
        matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=5,
            P1=8*3*window_size**2,
            P2=32*3*window_size**2,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            disp12MaxDiff=1,
            preFilterCap=63,
        )
    else:
        matcher = cv2.StereoBM_create(numDisparities=16*6, blockSize=15)

    disp = matcher.compute(grayL, grayR).astype(np.float32) / 16.0
    # normalize for visualization
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return disp, disp_vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    args = parser.parse_args()
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)
    disp, disp_vis = compute_disparity(left, right)
    cv2.imshow('disparity', disp_vis)
    cv2.waitKey(0)
