import glob
import numpy as np
import cv2 as cv
import os


# make sure to run 'cameraCalib.py' and 'arucoDetector.py' under the same camera:
def hand_eye_calib():
    # how many aruco pose is detected, list all of their dir:
    detected_aruco_ids = sorted(os.listdir("handEyeCalib/R_board2cam/"))

    # hand eye calibration (the result is R_cam2base, t_cam2base)
    R_ee2base = []
    t_ee2base = []
    R_target2cam = []
    t_target2cam = []

    for pose_id in detected_aruco_ids:
        # 1. get the R_gripper2base (R_base2gripper in eye-to-hand)
        # read T_gripper2base:
        T_gripper2base = np.loadtxt(f"handEyeCalib/kinematics/{pose_id}", delimiter=',')
        R_gripper2base = T_gripper2base[0:3, 0:3]
        t_gripper2base = T_gripper2base[0:3, -1] * 1000
        R_base2gripper = R_gripper2base.T # (3, 3)

        # 2. get the t_gripper2base (t_base2gripper in eye-to-hand)
        t_base2gripper = - R_base2gripper @ t_gripper2base # (3,)
        t_base2gripper = t_base2gripper[:, np.newaxis]

        # 3. get the R_target2cam
        R_board2camera = np.loadtxt(f"handEyeCalib/R_board2cam/{pose_id}", delimiter=',') # (3, 3)

        # 4. get the t_target2cam
        t_board2camera = np.loadtxt(f"handEyeCalib/t_board2cam/{pose_id}", delimiter=',') # (3,)
        t_board2camera = t_board2camera[:, np.newaxis]

        # append:
        R_ee2base.append(R_base2gripper)
        t_ee2base.append(t_base2gripper)
        R_target2cam.append(R_board2camera)
        t_target2cam.append(t_board2camera)

    R_cam2base, t_cam2base = cv.calibrateHandEye(R_ee2base, t_ee2base, R_target2cam, t_target2cam)
    return R_cam2base, t_cam2base


def hand_eye_test(imgs_dir: list, arucoDict, camera_mat, dist_coeff, R_cam2base, t_cam2base):

    # step 1: define a aruco detector:
    arucoParams = cv.aruco.DetectorParameters()
    arucoDictionary = cv.aruco.getPredefinedDictionary(arucoDict)
    arucoDetector = cv.aruco.ArucoDetector(arucoDictionary, arucoParams)

    # step 2: detect aruco
    for fname in imgs_dir:
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
            continue

        # step 3: for each images, we can detect the pose of each marker w.r.t the camera frame:
        if len(markerCorners) > 0:
            objectPoints = np.zeros((4, 3))
            objectPoints[0, 0:2] = (-5, 5) # unit in mm
            objectPoints[1, 0:2] = (5, 5)
            objectPoints[2, 0:2] = (5, -5)
            objectPoints[3, 0:2] = (-5, -5)
            objectPoints = objectPoints[:, :, np.newaxis]
            imagePoints = np.moveaxis(markerCorners[0], 0, -1)

            ret, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, camera_mat, dist_coeff)

            # draw the detected pose on the images:
            cv.aruco.drawDetectedMarkers(show_img, markerCorners, markerIds)
            cv.drawFrameAxes(show_img, camera_mat, dist_coeff, rvec, tvec, 10) # rvec and tvec is board2camera

            ############################
            # draw end axis in the frame:
            ############################
            # 1. calculate the inverse of hand-eye matrix:
            R_base2cam = R_cam2base.T
            t_base2cam = -R_base2cam @ t_cam2base
            T_base2cam = np.zeros((4, 4))
            T_base2cam[-1, -1] = 1
            T_base2cam[0:3, 0:3] = R_base2cam
            T_base2cam[0:3, -1] = t_base2cam.squeeze()

            # 2. for each forward kinematics reading get the R_gripper2base (R_base2gripper in eye-to-hand)
            T_gripper2base = np.loadtxt(f"handEyeCalib/kinematics/{fname.split('/')[-1].split('.')[0]}.csv", delimiter=',')
            T_gripper2base[0:3, -1] *= 1000

            # 3. get gripper to camera matrix
            T_gripper2camera = T_base2cam @ T_gripper2base
            rvec_gripper2camera, _ = cv.Rodrigues(T_gripper2camera[0:3, 0:3])
            tvec_gripper2camera = T_gripper2camera[0:3, -1]
            cv.drawFrameAxes(show_img, camera_mat, dist_coeff, rvec_gripper2camera, tvec_gripper2camera, 10)

            ############################
            ############################
            cv.imshow('aruco axis', show_img)
            cv.waitKey(0)
            if cv.waitKey(0) == ord('s'):
                cv.imwrite('test.png', show_img)
                
    cv.destroyAllWindows()




if __name__ == "__main__":
    # Step 1: hand-eye calibrate:
    R_cam2base, t_cam2base = hand_eye_calib()

    # Step 2: show detect marker axis (marker to camera) and the estimated ee frame (base to camera) in a single image
    camera_type = "left" # change to fit your case

    # step 1: define the used aruco marker info:
    arucoDict = cv.aruco.DICT_4X4_50
 
    # step 2: get all the images:
    imgs_dir = glob.glob(f"handEyeCalib/images/{camera_type}/*.png")
    imgs_dir.sort()

    # step3: get the intrisic matrix and distortion coefficient:
    cameraMatrix = np.loadtxt(f"camCalib/cameraModel/{camera_type}/intrinsic_mtx.csv", delimiter=',')
    distCoeffs = np.loadtxt(f"camCalib/cameraModel/{camera_type}/distortion.csv", delimiter=',')

    # step 4: detect aruco markers:
    hand_eye_test(imgs_dir, 
                  arucoDict,
                  cameraMatrix,
                  distCoeffs,
                  R_cam2base,
                  t_cam2base)