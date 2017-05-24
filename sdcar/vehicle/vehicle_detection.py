import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import numpy as np
import sys
import os
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label, center_of_mass, find_objects
from scipy.spatial.distance import cdist
from collections import deque

def _color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    takes in image bins and bins range and outputs individual channel histogram 
    and also combined histogram feature
    """
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def _bin_spatial(img, color_space=None, size=(32, 32)):
    """ convert image to new color spave and """
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    if color_space is not None:
        img = cv2.cvtColor(img, color_space)
    features = cv2.resize(img, size).ravel()
    # print(features.shape)
    # Return the feature vector
    return features

def _get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True, debug=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if vis is False:
        vis = debug

    if vis == True:
        features, hog_image = hog(img, \
            orientations=orient, \
            pixels_per_cell=(pix_per_cell, pix_per_cell), \
            cells_per_block=(cell_per_block, cell_per_block), \
            visualise=True, \
            feature_vector=False)
        if debug is True:
            plt.imshow(hog_image, cmap='gray')
            plt.title('hog image')
            plt.show()
        return features.ravel()
    else:
        features = hog(img, \
            orientations=orient, \
            pixels_per_cell=(pix_per_cell, pix_per_cell), \
            cells_per_block=(cell_per_block, cell_per_block), \
            visualise=False, \
            feature_vector=False)
        return features.ravel()

class VehicleDetectionClassifier():
    def __init__(self, force=False):
        self.epochs = 10
        self.X_scaler = None
        self.scaler_file = "./img_scaler.pkl"
        self.model = None
        self.modelfile = "./vehicle_classifier.pkl"
        if os.path.exists(self.modelfile) and not force:
            self._load_model()
        if os.path.exists(self.scaler_file) and not force:
            self._load_scaler()

    def _get_feature_scaler(self, feats):
        """ get feature scaler """
        self.X_scaler = StandardScaler()
        return self.X_scaler.fit(feats)

    def _extract_features(self, imgs, debug=False):
        """ extract features from list of string or list of images in RGB color space"""
        feats = []
        for img in imgs:
            if type(img) == str:
                if img.endswith("png") is True:
                    title = img
                    img = mpimg.imread(img)
                    if debug is True:
                        plt.imshow(img)
                        plt.title(title)
                        plt.show()
                    img = img*255.0
                else:
                    img = mpimg.imread(img)
            spatfeats = _bin_spatial(img)
            histfeats = _color_hist(img)
            hogfeats = _get_hog_features(img, debug=debug)
            feat = np.concatenate((spatfeats, histfeats, hogfeats), axis=0)
            feats.append(feat)
        return np.array(feats, dtype=np.float64)

    def extract_and_normalize_features(self, imgs, debug=False):
        """ extract and normalize features """
        feats = self._extract_features(imgs, debug=debug)
        if self.X_scaler is None:
            self.X_scaler = self._get_feature_scaler(feats)
            self._save_scaler()
        scaled_feats = self.X_scaler.transform(feats)
        return scaled_feats

    def predict(self, img, debug=False):
        """ pass a 64x64 image and return True or False"""
        feats = self.extract_and_normalize_features([img], debug=debug)
        y = self.model.predict(feats)
        return y

    def train(self, vehicle_images, nonvehicle_images, overwrite=False):
        if not overwrite and self.model is not None:
            return

        data_files = vehicle_images + nonvehicle_images
        scaled_X = self.extract_and_normalize_features(data_files)
        y = np.hstack((np.ones(len(vehicle_images)), np.zeros(len(nonvehicle_images))))

        if self.model is None:
            if False:
                parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
                svr = SVC()
                clf = GridSearchCV(svr, parameters)
                clf.fit(scaled_X, y)
                print(clf.best_params_)
                # grid searched SVC returns C = 10 and kernel 'rbf'
            else:
                clf = LinearSVC()

            self.model = clf

        for n in range(self.epochs):
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
            # split data into training and testing
            self.model.fit(X_train, y_train)
            print("Accuracy {:.3f}", self.model.score(X_test, y_test))
        self._save_model()
        return
        
    def _save_scaler(self):
        joblib.dump(self.X_scaler, self.scaler_file)
        return

    def _load_scaler(self):
        self.X_scaler = joblib.load(self.scaler_file)
        return

    def _save_model(self):
        joblib.dump(self.model, self.modelfile)
        return

    def _load_model(self):
        self.model = joblib.load(self.modelfile)
        return
    
    # Define a function to return some characteristics of the dataset 
    def _data_look(self, vehicle_images, nonvehicle_images):
        data_dict = {}
        # Define a key in data_dict "n_cars" and store the number of car images
        data_dict["n_cars"] = len(vehicle_images)
        # Define a key "n_notcars" and store the number of notcar images
        data_dict["n_notcars"] = len(nonvehicle_images)
        carimg = mpimg.imread(vehicle_images[0])
        # Read in a test image, either car or notcar
        # Define a key "image_shape" and store the test image shape 3-tuple
        data_dict["image_shape"] = carimg.shape
        # Define a key "data_type" and store the data type of the test image.
        data_dict["data_type"] = carimg.dtype
        # Return data_dict
        return data_dict
    
class VehicleBox():
    def __init__(self, N, position, size):
        """ Once a vehicle appears in N frames becomes eligible"""
        self.threshold = N/2
        self.positions = deque([], maxlen=N)
        self.sizes = deque([], maxlen=N)
        self.positions.append(position)
        self.sizes.append(size)
    
    def computation_possible(self):
        return (len(self.positions) >= self.threshold)

    def distance(self, new_point, cur_point = None):
        if cur_point is None:
            cur_point = self.average_position()
        distance = cdist(np.array([cur_point]).astype(np.uint32), np.array([new_point]).astype(np.uint32))
        return distance

    def average_position(self):
        if len(self.positions) == 0:
            return None
        position = np.mean(self.positions, axis=0)
        return position
    
    def average_size(self):
        if len(self.sizes) == 0:
            return None
        size = np.mean(self.sizes, axis=0)
        return size

    def add(self, positions, sizes):
        # find a position closest to this vehicle
        # self.positions.put(position)
        # self.sizes.put(sizes)
        cur_position = self.average_position()
        # print(cur_position)
        templist = [(x, y, self.distance(x, cur_point=cur_position)) for x, y in zip(positions, sizes)]
        if len(templist) == 0:
            self.positions.popleft()
            self.sizes.popleft()
            return None

        minimum_dist = min(templist, key=lambda x: x[2])
        # print(minimum_dist)
        if minimum_dist[1][0] < 96:
            similarity_pixels = 50
        else:
            similarity_pixels = 100

        if minimum_dist[2] < similarity_pixels:
            self.positions.append(minimum_dist[0])
            self.sizes.append(minimum_dist[1])
        else:
            self.positions.popleft()
            self.sizes.popleft()
        return minimum_dist[0], minimum_dist[1]

    def get_box(self):
        # calculate box from positions and sizes
        position = np.int32(self.average_position())
        size = np.int32(self.average_size())
        length_2 = np.int32(size[0]/2)
        width_2 = np.int32(size[1]/2)
        return (((position[1] - width_2, position[0] - length_2), (position[1] + width_2, position[0] + length_2)), self.computation_possible())


class DetectedVehiclesTracker():
    def __init__(self, tracking_frames=6):
        self.vehicles = []
        self.tracking_frames = tracking_frames

    def add_new_detections(self, positions, boxes, objects):
        # print(objects)
        if len(positions) == 0:
            return

        sizes = [ (abs(obj[0].start - obj[0].stop), abs(obj[1].start - obj[1].stop)) for obj in objects ]
        for vehicle in self.vehicles:
            retval = vehicle.add(positions, sizes)
            if retval is None:
                continue
            pos = retval[0]
            size = retval[1]
            positions.remove(pos)
            sizes.remove(size)

        # create new detections
        for pos, size in zip(positions, sizes):
            self.vehicles.append(VehicleBox(self.tracking_frames, pos, size))

    def get_boxes(self):

        # remove idle detections
        self.vehicles = [ x for x in self.vehicles if len(x.positions) > 0 ]

        # fetch boxes for remaining detections
        boxes = [ x.get_box() for x in self.vehicles ]
        
        # filter to only active detections
        boxes = [ box[0] for box in boxes if box[1] is True ]
        return boxes


class VehicleDetectionPipeline():
    """This class is used to detect other cars on the road"""
    def __init__(self, clf, tracking_frames=6):
        self.clf = clf
        self.tracker = DetectedVehiclesTracker(tracking_frames=tracking_frames)
        
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def slide_window(self, img, xboundary=[None, None], yboundary=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        if xboundary[0] is None:
            xboundary[0] = 0
        if xboundary[1] is None:
            xboundary[1] = img.shape[1]
        if yboundary[0] is None:
            yboundary[0] = 0
        if yboundary[1] is None:
            yboundary[1] = img.shape[0]
            
        xy_span = (xboundary[1] - xboundary[0], yboundary[1] - yboundary[0])
        xy_pixels = (np.int((1-xy_overlap[0])*xy_window[0]), np.int((1-xy_overlap[1])*xy_window[1]))

        # Initialize a list to append window positions to
        window_list = []
        for x in range(xboundary[0], xboundary[1], xy_pixels[0]):
            for y in range(yboundary[0], yboundary[1], xy_pixels[1]):
                top_left = (x, y)
                bottom_right = ((x+xy_window[0]),(y + xy_window[1]))
                window_list.append((top_left, bottom_right))
        # Return the list of windows
        return window_list
    
    def search_windows(self, img, windows, debug=False):
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            prediction = self.clf.predict(test_img)
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
    
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, bboxes, color=(0,0,255)):
        # Iterate through all detected cars
        for bbox in bboxes:
            # # Find pixels with each car_number label value
            # nonzero = (labels[0] == car_number).nonzero()
            # # Identify x and y values of those pixels
            # nonzeroy = np.array(nonzero[0])
            # nonzerox = np.array(nonzero[1])
            # # Define a bounding box based on min/max x and y
            # bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], color, 6)

        # Return the image
        return img

    def track_labels(self, heatmap, labels):
        # find center of mass of labels
        objs = find_objects(labels[0])
        boxes = []
        positions = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            position = center_of_mass((labels[0] == car_number), labels=[car_number])
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            boxes.append(bbox)
            positions.append(position)

        self.tracker.add_new_detections(positions, boxes, objs)

        return positions, objs, boxes


    def run(self, img, debug=False, save_output=None):
        """return image with detected cars in the dashcam image"""
        # RGB image
        if debug is True:
            plt.imshow(img)
            plt.title("input image")
            plt.show()

        if save_output is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_output + "input.jpg", img)
        
        # slide windows of different sizes
        windows = []
        for pixels in range(32, 256, 32):
            scale = pixels/float(256.0)
            yboundary = [360, np.max([np.int(360 + 360*scale)])]
            if pixels < 96:
                overlap = (0,0)
            else:
                overlap = (0.5, 0.5)
            windows_perscale = self.slide_window(img, xboundary=[0, 1280], yboundary=yboundary,
                        xy_window=(pixels, pixels), xy_overlap=overlap)
            windows += windows_perscale

        if debug is True:
            copy = self.draw_boxes(img, windows)
            plt.imshow(copy)
            plt.title("slide windows")
            plt.show()

        # search windows using the output of slide windows
        hot_windows = self.search_windows(img, windows)
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = self.add_heat(heat, hot_windows)

        # apply threshold
        heat = self.apply_threshold(heat, 1)
        heatmap = np.clip(heat, 0, 255)
        if debug is True:
            plt.imshow(heatmap, cmap='hot')
            plt.title('heatmap')
            plt.show()

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        positions, objs, boxes = self.track_labels(heatmap, labels)
        bboxes = self.tracker.get_boxes()
        if debug is True:
            print(positions)
            print("Boxes:", boxes)
            print("Tracked:", bboxes)
            print(objs)
            print(labels)
        
        final_img = self.draw_labeled_bboxes(np.copy(img), bboxes)
        # final_img = self.draw_labeled_bboxes(np.copy(final_img), bboxes, color=(255,0,0))
        # create heat map
        if debug is True:
            plt.imshow(final_img)
            plt.title("processed image")
            plt.show()

        # labels
        return final_img


if __name__ == "__main__":
    fnames = glob.glob("../../test_images/test*.jpg")
    fnames = fnames[:]

    import argparse
    parser = argparse.ArgumentParser(description='Test Pipeline')
    parser.add_argument('output_dir', nargs='?', type=str, help='output image directory', default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.add_argument('--train', dest='model', action='store_true')
    parser.add_argument('--no-train', dest='model', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    args = parser.parse_args()
    debug = args.debug
    model_train = args.model
    save_output_dir = args.output_dir
    save = args.save


    # print(model_train)
    classifier = VehicleDetectionClassifier(force=model_train)
    if args.save is True:
        vehicles = glob.glob("../../vehicles/*/*png")
        nonvehicles = glob.glob("../../non-vehicles/*/*png")
        classifier.extract_and_normalize_features(vehicles[10:11], debug=True)
        classifier.extract_and_normalize_features(nonvehicles[10:11], debug=True)
        os.Exit(0)
        

    if model_train is True:
        vehicles = glob.glob("../../vehicles/*/*png")
        nonvehicles = glob.glob("../../non-vehicles/*/*png")
        output = classifier._data_look(vehicles, nonvehicles)
        print(output)
        classifier.train(vehicles, nonvehicles, overwrite=True)

    # print(save_output_dir)
    pipeline = VehicleDetectionPipeline(classifier, tracking_frames = 2)

    # img = mpimg.imread("../../vehicles/GTI_Far/image0069.png")*255.0
    # print(img)
    fnames = fnames[0:5]
    for fname in fnames:
        img = mpimg.imread(fname)
        pipeline.run(img, debug=debug, save_output=save_output_dir)