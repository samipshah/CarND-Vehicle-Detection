**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/car_hog.png
[image3]: ./output_images/noncar.png
[image4]: ./output_images/noncar_hog.png
[image5]: ./output_images/sliding_window.png
[image6]: ./output_images/input_heat.png
[video1]: ./project_output.mp4
[video2]: ./test_output.mp4

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In `sdcar/vehicle/vehicle_detection.py` file there is a function called `_get_hog_features` which extracts hog features from image section.

I started with classification part first in VehicleDetectionClassifier. That class given an image it classifies it's whether an image of car or not.
I started with hog feature function written in some of the other exercise and used visualization to see if hog features are figuring out proper shapes of different parts of cars.



I also explored a bunch of color spaces , as you could see that from `sdcar/vehicle/helper.py`. In that script I plotted in 3D different images in varied color scheme to figure out 

In my final solution I have used `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![car][image1] | ![car hog features][image2]

![non car][image3] | ![non car hog features][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried some parameters however, I did not tune a lot of parameters as my classifier was already giving a very good accuracy on testing dataset

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a LinearSVC in my `VideoDetectionClassifier`. I also tried grid search to find the perfect parameter for `SVC` and it came out with C=10 and kernel='rbf'. And I noticed that the testing set accuracy was nearly 1. However, I noticed classic example of overfitting with it. It did not work well on the actual video. Hence I resorted to using LinearSVC which worked quite well.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I took the sliding window approach with scaling. Based on the sliding window dimension I decide how much to overlap. I used windows of  size 32, 64, 96, 128, 160, 192, 224, 256. And used overlap of 0 if less than 96 and overlap of 0.5 if greater. This may not really be optimum but it is derived from the fact in case of larger windows if I do not have overlap , I can miss bigger and probably closer car images. Moreover number of windows reduces drastically when reducing overlap for smaller window sizes

![sliding window][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For classifier's performance I basically used Hog features over gray channel and then combined that with spatial binning with 32 x 32 window and also color histogram  with 32 bins and all three channels.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

![heat map][image]

I also tried tracking the boxes , however I have implemented very basic tracking of centroid. I basically track past 10 boxes in the vicinity and vicinity I measure currently if it is with 100 euclidean pixel distance away if the box is bigger and 50 if the boxes are smaller. Certainly that logic can be improved upon. However, this logic certainly helps in avoiding false positives.


---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think this can be improved radically by using tracking pipeline properly and also chosing sliding windows efficiently. We could track direction and speed in tracking vehicle. Also using that information we can chose sliding windows more efficiently. Also current implementation relies on RGB color space and may be some other colorspace would give a better output. Other issue I see is slope on the road. If slope is there I can not keep the scanning window logic limited to 360-720 it needs to change with slope.