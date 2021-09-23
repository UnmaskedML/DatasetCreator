from mjpegstreamer import MJPEGServer
from openimages import OpenImagesDownloader
from glob import glob
import cv2
import pandas as pd
import threading


class Mask:
    """
    class for dealing with masks
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        self.height = self.y_max - self.y_min
        self.width = self.x_max - self.x_min
        self.vertical_scale = 2
        self.orientations = ["mid", "left", "right", "mid_left", "mid_right"]
        self.orientation = 0
        self.mask_dir = "./data/images/masks"

    def get_mask(self):
        filename = self.orientations[self.orientation] + "_blue.png"
        mask = cv2.imread(self.mask_dir + "/" + filename, cv2.IMREAD_UNCHANGED)
        print("shape:", mask.shape)
        resized_mask = cv2.resize(mask, (self.width, self.height // self.vertical_scale))
        return resized_mask

    def next(self):
        self.orientation += 1
        if self.orientation > len(self.orientations):
            self.orientation = 0


class NormalFaceImage:
    def __init__(self, path, normal_labels):
        self.filename = path.split('/')[-1]
        self.normal_labels = normal_labels
        self.image = cv2.imread(path)

    def get_labels(self):
        return self.normal_labels.loc[self.normal_labels["key"] == self.filename]

    def test_mask(self, mask):
        tmp_image = self.image
        mask_image = mask.get_mask()
        for y in range(mask.height//mask.vertical_scale):
            for x in range(mask.width):
                pixel = mask_image[y, x]
                if pixel[3] != 0:
                    tmp_image[mask.y_min+mask.height-mask.height//mask.vertical_scale, mask.x_min+x] = pixel[0:3]
        return tmp_image


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
        self.normal_image_paths = glob(self.normal_dir + "/*.jpg")
        self.normal_labels = pd.read_csv(self.labels_dir + "/normal_faces.csv")
        self.server = MJPEGServer()
        self.server.send_image(cv2.imread(self.normal_image_paths[0]))
        server_thread = threading.Thread(target=self.server.start, args=(4000,))
        server_thread.start()

    def download_normal_faces(self):
        downloader = OpenImagesDownloader(5000)
        downloader.download_normal_faces()
        downloader.create_normal_csv()
        downloader.clean()
        self.normal_image_paths = glob(self.normal_dir + "/*.jpg")

    def mask(self):
        for path in self.normal_image_paths:
            normal_face = NormalFaceImage(path, self.normal_labels)
            labels = normal_face.get_labels()
            print(labels, type(labels))
            i, label = [*labels.iterrows()][1]
            print(label)
            mask = Mask(label["xmin"], label["xmax"], label["ymin"], label["ymax"])
            self.server.send_image(normal_face.test_mask(mask))
            break


if __name__ == "__main__":
    photoshop = Photoshop()
    photoshop.mask()
