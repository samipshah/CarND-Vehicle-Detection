import matplotlib.pyplot as plt

class VehicleDetectionPipeline():
    """This class is used to detect other cars on the road"""
    def __init__(self):
        pass

    def run(self, img, debug=False):
        """return image with detected cars in the dashcam image"""
        if debug is True:
            plt.imshow(img)
            plt.title("input image")
            plt.show()

        return img
