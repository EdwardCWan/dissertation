import cv2 as cv
import numpy as np
import argparse
import glob


def camera_calibration(all_imgs_dir:list, 
                       num_imgs_used: int, 
                       corners_x: int,
                       corners_y: int,
                       square_size: float,
                       verbose: bool):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Step 1: detect corners
    # prepare the object points in 3D w.r.t the {board} frame:
    objp = np.zeros((corners_x * corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    detected_img = []

    for fname in all_imgs_dir[0:num_imgs_used]:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (corners_x, corners_y), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            detected_img.append(fname)
            objpoints.append(objp)
            # refine the corner to make it more accurate:
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            if verbose:
                cv.drawChessboardCorners(img, (corners_x, corners_y), corners2, ret)
                cv.imshow('img', img)
                # cv.waitKey(0)
                if cv.waitKey(0) ==ord('s'):
                    cv.imwrite('test.png', img)
        else:
            print(f"{fname} fail to detect require chessboard pattern.")
    cv.destroyAllWindows()

    # step 2: calibrate the camera
    # image size is (1920, 1080) = (size_in_x_dir, size_in_y_dir)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    result = mtx.copy(), dist.copy()

    # step 4: check the accuracy of the camera calibration using reprojection error:
    mean_error = 0
    for i, fname in enumerate(detected_img):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        print(f"{fname} has the reprojection error: {error:.6f}")
        mean_error += error
    print("The error is the difference between detect corners and reprojected corners in the images.")
    print("The closer to zero, the better camera calibration performance.")
    print("Total error (in pixels): {}".format(mean_error/len(objpoints)))

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument("-n", "--num_imgs_used", type=int, required=True)
    parser.add_argument("-t", "--camera_type", type=str, required=True) # left/right
    # chessboard info:
    parser.add_argument("-cx", "--corners_x", type=int, required=True)
    parser.add_argument("-cy", "--corners_y", type=int, required=True)
    parser.add_argument("-s", "--square_size", type=float, required=True)
    parser.add_argument("-v", "--verbose", type=int, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    import os
    import shutil

    args = parse_args()

    # if images folder exist, rm and mkdir a empty one:
    if os.path.exists("camCalib/cameraModel"):
        shutil.rmtree("camCalib/cameraModel")
        os.mkdir("camCalib/cameraModel")
        os.makedirs("camCalib/cameraModel/left")
        os.makedirs("camCalib/cameraModel/right")
    else:
        os.mkdir("camCalib/cameraModel")
        os.makedirs("camCalib/cameraModel/left")
        os.makedirs("camCalib/cameraModel/right")

    # Step 1: get all images dir:
    imgs_dir = glob.glob(f"camCalib/images/{args.camera_type}/*.png")
    imgs_dir.sort()

    # Step 2: camera calibration -> return the intrinsic matrix of the camera model: (from camera to image)
    # intrinsic_r, intrinsic_t = 
    intrinsic_mtx, distortion_params = camera_calibration(
        all_imgs_dir=imgs_dir,
        num_imgs_used=args.num_imgs_used,
        corners_x=args.corners_x,
        corners_y=args.corners_y,
        square_size=args.square_size, # interms of mm
        verbose=args.verbose,
    )

    # step 3: save the camera matrix and distortion params:
    np.savetxt(f"camCalib/cameraModel/{args.camera_type}/intrinsic_mtx.csv", intrinsic_mtx, delimiter=',')
    np.savetxt(f"camCalib/cameraModel/{args.camera_type}/distortion.csv", distortion_params, delimiter=',')