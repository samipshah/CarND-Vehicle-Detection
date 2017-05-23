import sys
from sdcar.vehicle.vehicle_detection import VehicleDetectionPipeline
from sdcar.vehicle.vehicle_detection import VehicleDetectionClassifier
import matplotlib.pyplot as plt
import numpy as np
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
classifier = VehicleDetectionClassifier()
vehicle_detect = VehicleDetectionPipeline(classifier)
# read a video
cap = cv2.VideoCapture(video_file)
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outframe = vehicle_detect.run(frame, debug=debug)
    outframe = cv2.cvtColor(outframe, cv2.COLOR_RGB2BGR)
    byteframe = outframe.astype('uint8')
    cv2.imwrite("output/IMG/" + str(i) + ".jpg", byteframe)
    i += 1
    
cap.release()
cv2.destroyAllWindows()