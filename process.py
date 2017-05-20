import sys
from sdcar.vehicle.vehicle_detection import VehicleDetectionPipeline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

# process arguments
import argparse
parser = argparse.ArgumentParser(description='Vehicle Detect Pipeline')
parser.add_argument('video', nargs='?', type=str, help='video to process', default="test_video.mp4")
parser.add_argument('--d', dest='debug', type=bool, default=False, help='debug flag')
args = parser.parse_args()
debug = args.debug
video_file = args.video

# create pipeline
vehicle_detect = VehicleDetectionPipeline()
# read a video
cap = cv2.VideoCapture(video_file)
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    outframe = vehicle_detect.run(frame, debug=debug)
    byteframe = outframe.astype('uint8')
    cv2.imwrite("output/IMG/" + str(i) + ".jpg", byteframe)
    i += 1
    
cap.release()
cv2.destroyAllWindows()