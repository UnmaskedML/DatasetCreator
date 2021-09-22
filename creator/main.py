from openimages import OpenImagesDownloader
from glob import glob
import cv2


class Photoshop:
    """
    class for editting masks onto faces, and saving important data
    """

    def __init__(self):
        self.data_dir = "./data"
        self.labels_dir = self.data_dir + "/labels"
        self.images_dir = self.data_dir + "/images"
        self.normal_dir = self.images_dir + "/normal_faces"  # unmasked faces
        self.masks_dir = self.images_dir + "/masks"
        self.mask_dir = self.images_dir + "/masked_faces"
        self.images = glob(self.images_dir + "/*.jpg")


if __name__ == "__main__":
    downloader = OpenImagesDownloader(5000)
    downloader.download_normal_faces()
    downloader.create_normal_csv()
    downloader.clean()
