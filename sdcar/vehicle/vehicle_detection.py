import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import sys

class VehicleDetectionPipeline():
    """This class is used to detect other cars on the road"""
    def __init__(self):
        pass

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

        return img

if __name__ == "__main__":
    pipeline = VehicleDetectionPipeline()
    fnames = glob.glob("../../test_images/test*.jpg")
    fnames = fnames[:]

    import argparse
    parser = argparse.ArgumentParser(description='Test Pipeline')
    parser.add_argument('output_dir', nargs='?', type=str, help='output image directory', default=None)
    parser.add_argument('--d', dest='debug', type=bool, default=False, help='debug flag')
    args = parser.parse_args()
    debug = args.debug
    save_output_dir = args.output_dir

    print(save_output_dir)
    for fname in fnames:
        img = mpimg.imread(fname)
        pipeline.run(img, debug=debug, save_output=save_output_dir)



