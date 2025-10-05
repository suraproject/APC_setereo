"""Stereo calibration helper

Collect chessboard pairs from two cameras, run stereoCalibration, and save parameters.
"""
import cv2
import numpy as np
import argparse
import os

def collect_chessboards(output_dir, cam_index=0, board_size=(9,6)):
    cap = cv2.VideoCapture(cam_index)
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    pattern = board_size
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        draw = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(draw, pattern, corners, found)
        cv2.imshow(f"cam{cam_index} capture", draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and found:
            fname = os.path.join(output_dir, f"{cam_index}_{count:03d}.png")
            cv2.imwrite(fname, frame)
            print("Saved:", fname)
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def stereo_calibrate(left_images, right_images, board_size=(9,6), square_size=1.0, out_file='stereo_params.npz'):
    # prepare object points
    objp = np.zeros((board_size[0]*board_size[1],3), np.float32)
    objp[:,:2] = np.indices(board_size).T.reshape(-1,2)
    objp *= square_size

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    for l, r in zip(left_images, right_images):
        il = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
        if il is None or ir is None:
            continue
        found_l, corners_l = cv2.findChessboardCorners(il, board_size, None)
        found_r, corners_r = cv2.findChessboardCorners(ir, board_size, None)
        if found_l and found_r:
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

    if not objpoints:
        raise RuntimeError('No valid chessboard pairs found')

    # image size from first left image
    img_shape = il.shape[::-1]

    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    flags = cv2.CALIB_FIX_INTRINSIC
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, img_shape, criteria=stereocalib_criteria, flags=flags
    )

    print('Stereo calibrate RMS:', ret)
    np.savez(out_file, M1=M1, d1=d1, M2=M2, d2=d2, R=R, T=T, E=E, F=F)
    print('Saved stereo params to', out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_dir', required=True)
    parser.add_argument('--right_dir', required=True)
    parser.add_argument('--out', default='stereo_params.npz')
    args = parser.parse_args()

    left_imgs = sorted([os.path.join(args.left_dir, f) for f in os.listdir(args.left_dir)])
    right_imgs = sorted([os.path.join(args.right_dir, f) for f in os.listdir(args.right_dir)])
    stereo_calibrate(left_imgs, right_imgs, out_file=args.out)
