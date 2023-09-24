import cv2 as cv
import numpy as np
import argparse
import glob

def aruco_detect(imgs_dir: list, arucoDict, camera_mat, dist_coeff):

    # step 1: define a aruco detector:
    arucoParams = cv.aruco.DetectorParameters()
    arucoDictionary = cv.aruco.getPredefinedDictionary(arucoDict)
    arucoDetector = cv.aruco.ArucoDetector(arucoDictionary, arucoParams)

    # step 2: detect aruco
    for i in range(len(imgs_dir)):
        fname = imgs_dir[i]
        # read image:
        img = cv.imread(fname)
        show_img = img.copy()

        # change to gray scale:
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # In one single image, find the aruco markers, return:    
        # - corners (tuple): each entries contains a 4x2 matrix,
        #   which is the pixel position of 4 corners of the markers.
        # - markerIds (np.array/None): all detected markers id, it can detect multiple marker simultaneously
        # - rejected (tuple): each entries is a 4x2 matrix, it is the detected square that is not a aruco marker
        markerCorners, markerIds, rejected = arucoDetector.detectMarkers(grey)
        # print out a message, when no aruco is detected on the image:
        if not markerIds: # markerIds is None when no markers detected in the current image
            print(f"{fname} has no aruco markers detected!")


        # step 3: for each images, we can detect the pose of each marker w.r.t the camera frame:
        if len(markerCorners) > 0:
            # for j in range(0, len(markerIds)):
            # rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(markerCorners[j], 0.01, camera_mat, dist_coeff)
            objectPoints = np.zeros((4, 3))
            objectPoints[0, 0:2] = (-5, 5) # unit in mm
            objectPoints[1, 0:2] = (5, 5)
            objectPoints[2, 0:2] = (5, -5)
            objectPoints[3, 0:2] = (-5, -5)
            objectPoints = objectPoints[:, :, np.newaxis]

            imagePoints = np.moveaxis(markerCorners[0], 0, -1)

            # draw the detected pose on the images:
            cv.aruco.drawDetectedMarkers(show_img, markerCorners, markerIds)

            ret, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, camera_mat, dist_coeff)
            if ret:
                cv.drawFrameAxes(show_img, camera_mat, dist_coeff, rvec, tvec, 10, thickness=4) # unit in mm

            cv.imshow('aruco axis', show_img)

            if cv.waitKey(0) == ord('s'):
                # save the estimated marker pose:
                # 1. use rodrigues to get rotation matrix:
                R_marker2cam = rvec.copy()
                t_marker2cam = tvec.copy()
                cv.imwrite('test.png', show_img)

                np.savetxt(f"handEyeCalib/R_board2cam/{fname.split('/')[-1].split('.')[0]}.csv", R_marker2cam, delimiter=',') # 3,
                np.savetxt(f"handEyeCalib/t_board2cam/{fname.split('/')[-1].split('.')[0]}.csv", t_marker2cam, delimiter=',') # 3,
    
    cv.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument("-t", "--camera_type", type=str, required=True) # left/right
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    import shutil
    import os
    if not os.path.exists("handEyeCalib/R_board2cam"):
        os.mkdir("handEyeCalib/R_board2cam")
    else:
        shutil.rmtree("handEyeCalib/R_board2cam")
        os.mkdir("handEyeCalib/R_board2cam")

    if not os.path.exists("handEyeCalib/t_board2cam"):
        os.mkdir("handEyeCalib/t_board2cam")
    else:
        shutil.rmtree("handEyeCalib/t_board2cam")
        os.mkdir("handEyeCalib/t_board2cam")



    # get the input arguments from command line:
    args = parse_args()

    # step 1: define the used aruco marker info:
    arucoDict = cv.aruco.DICT_4X4_50

    # step 2: get all the images:
    imgs_dir = glob.glob(f"handEyeCalib/images/{args.camera_type}/*.png")
    imgs_dir.sort()

    # step3: get the intrisic matrix and distortion coefficient:
    cameraMatrix = np.loadtxt(f"camCalib/cameraModel/{args.camera_type}/intrinsic_mtx.csv", delimiter=',')
    distCoeffs = np.loadtxt(f"camCalib/cameraModel/{args.camera_type}/distortion.csv", delimiter=',')

    # step 4: detect aruco markers:
    aruco_detect(imgs_dir, 
                 arucoDict,
                 cameraMatrix,
                 distCoeffs)