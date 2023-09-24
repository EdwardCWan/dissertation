import cv2 as cv
import glob
import os
import shutil

dir = "camCalib/videos/*.avi"

# if images folder exist, rm and mkdir a empty one:
if os.path.exists(f"{dir.split('/')[0]}/images"):
    shutil.rmtree(f"{dir.split('/')[0]}/images")
    os.mkdir(f"{dir.split('/')[0]}/images")
    os.makedirs(f"{dir.split('/')[0]}/images/left")
    os.makedirs(f"{dir.split('/')[0]}/images/right")
else:
    os.mkdir(f"{dir.split('/')[0]}/images")
    os.makedirs(f"{dir.split('/')[0]}/images/left")
    os.makedirs(f"{dir.split('/')[0]}/images/right")

all_video_dirs = glob.glob(dir)

for fname in all_video_dirs:
    camera_type = fname.split('.')[0].split('_')[-1]
    frame_id = fname.split('/')[-1].split('_')[0]

    save_dir = f"{dir.split('/')[0]}/images/{camera_type}/{frame_id}.png"

    # play the video:
    cap = cv.VideoCapture(fname)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("get frame fail or stream end!")
            break

        cv.imshow("img", frame)
        if cv.waitKey(0) == ord('s'):
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            print(save_dir)
            cv.imwrite(save_dir, frame)
            break

    cap.release()
cv.destroyAllWindows()
