#!/usr/bin/env python

# Author: Anton Deguet
# Date: 2015-02-22

# (C) Copyright 2015-2021 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

# Start a single arm using
# > rosrun dvrk_robot dvrk_console_json -j <console-file>

# To communicate with the arm using ROS topics, see the python based example dvrk_arm_test.py:
# > rosrun dvrk_python dvrk_arm_test.py <arm-name>

import dvrk
import math
import sys
import time
import rospy
import numpy
import PyKDL
import argparse
import cv2 as cv

# print with node id
def print_id(message):
    print('%s -> %s' % (rospy.get_caller_id(), message))

# example of application using arm.py
class example_application:

    # configuration
    def configure(self, robot_name, expected_interval):
        print_id('configuring dvrk_arm_test for %s' % robot_name)
        self.expected_interval = expected_interval
        self.arm = dvrk.arm(arm_name = robot_name,
                            expected_interval = expected_interval)


    # homing example
    def home(self):
        print_id('starting enable')
        if not self.arm.enable(10):
            sys.exit('failed to enable within 10 seconds')
        print_id('starting home')
        if not self.arm.home(10):
            sys.exit('failed to home within 10 seconds')
        # get current joints just to set size
        print_id('move to starting position')
        goal = numpy.copy(self.arm.setpoint_jp())
        # go to zero position, for PSM and ECM make sure 3rd joint is past cannula
        goal.fill(0)
        if ((self.arm.name() == 'PSM1') or (self.arm.name() == 'PSM2')
            or (self.arm.name() == 'PSM3') or (self.arm.name() == 'ECM')):
            goal[2] = 0.12
        # move and wait
        print_id('moving to starting position')
        self.arm.move_jp(goal).wait()
        # try to move again to make sure waiting is working fine, i.e. not blocking
        print_id('testing move to current position')
        move_handle = self.arm.move_jp(goal)
        time.sleep(1.0) # add some artificial latency on this side
        move_handle.wait()
        print_id('home complete')
        print_id(self.arm.measured_jp())


    def kin_monitor(self):

        # oepn the camera:
        cap = cv.VideoCapture(0)

        # font:
        font = cv.FONT_HERSHEY_SIMPLEX
        
        # save frame:
        save_frame_id = 1

        while cap.isOpened():
        # read one frame:
            ret, frame = cap.read()
            ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            if not ret:
                print_id("can't receive frame (stream end?). Exiting ...")
                break

            # read one joint state:
            current_jp = self.arm.measured_jp()
            k1 = math.degrees(current_jp[0])
            k2 = math.degrees(current_jp[1])
            k3 = current_jp[2]
            k4 = math.degrees(current_jp[3])
            k5 = math.degrees(current_jp[4])
            k6 = math.degrees(current_jp[5])

            # draw on the frame:
            cv.putText(frame, f"joint1: {int(k1):d}", (440, 100), font, 0.8, (0,100,255), 2, cv.LINE_AA)
            cv.putText(frame, f"joint2: {int(k2):d}", (440, 140), font, 0.8, (0,100,255), 2, cv.LINE_AA)
            cv.putText(frame, f"joint3: {k3:.3f}", (440, 180), font, 0.8, (0,100,255), 2, cv.LINE_AA)
            cv.putText(frame, f"joint4: {int(k4):d}", (440, 220), font, 0.8, (0,100,255), 2, cv.LINE_AA)
            cv.putText(frame, f"joint5: {int(k5):d}", (440, 260), font, 0.8, (0,100,255), 2, cv.LINE_AA)
            cv.putText(frame, f"joint6: {int(k6):d}", (440, 300), font, 0.8, (0,100,255), 2, cv.LINE_AA)

            # show:
            cv.imshow('frame', frame)
            
            # sleep for certain time, the video is showed 30 fps
            if cv.waitKey(1) == ord('s'):
                jp_reading = numpy.copy(self.arm.measured_jp())
                numpy.savetxt(f"{str(save_frame_id).zfill(2)}.csv", jp_reading, delimiter=',')
                print_id(f"Frame {save_frame_id}'s joint state has been saved!")
                save_frame_id += 1
      

    # main method
    def run(self):
        self.home()
        self.kin_monitor()


if __name__ == '__main__':
    # ros init node so we can use default ros arguments (e.g. __ns:= for namespace)
    rospy.init_node('dvrk_arm_test', anonymous=True)
    # strip ros arguments
    argv = rospy.myargv(argv=sys.argv)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True,
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help = 'arm name corresponding to ROS topics without namespace.  Use __ns:= to specify the namespace')
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help = 'expected interval in seconds between messages sent by the device')
    args = parser.parse_args(argv[1:]) # skip argv[0], script name

    application = example_application()
    application.configure(args.arm, args.interval)
    application.run()
