from mjpegstreamer import MJPEGServer
from openimages import OpenImagesDownloader
from glob import glob
import cv2
import pandas as pd
import threading
import numpy as np


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
        self.colors = ["black", "white", "blue"]
        self.color = 0

    def get_mask(self):
        filename = self.orientations[self.orientation] + "_{}.png".format(self.colors[self.color])
        mask = cv2.imread(self.mask_dir + "/" + filename, cv2.IMREAD_UNCHANGED)
        resized_mask = cv2.resize(mask, (self.width, self.height // self.vertical_scale))
        return resized_mask

    def next_orientation(self):
        self.orientation += 1
        if self.orientation >= len(self.orientations):
            self.orientation = 0

    def next_color(self):
        self.color += 1
        if self.color >= len(self.colors):
            self.color = 0

    def get_coordinates(self):
        """
        :return: Coordinates of this mask
        """
        return self.x_min, self.y_max - self.height // self.vertical_scale, self.y_max, self.y_max


class NormalFaceImage:
    def __init__(self, path, normal_labels):
        self.filename = path.split('/')[-1]
        self.normal_labels = normal_labels
        self.image = cv2.imread(path)

    def get_labels(self):
        """
        :return: labels from csv file for this image
        """
        return self.normal_labels.loc[self.normal_labels["key"] == self.filename]

    def test_mask(self, mask: Mask):
        """
        Tests a mask on this face
        :param mask: a mask object
        :return: an OpenCV image
        """
        tmp_image = np.copy(self.image)
        mask_image = mask.get_mask()
        for y in range(mask.height // mask.vertical_scale):  # mask covers bottom half of face
            for x in range(mask.width):
                pixel = mask_image[y, x]
                if pixel[3] != 0:  # checks alpha channel
                    tmp_y = mask.y_min + mask.height - mask.height // mask.vertical_scale + y
                    tmp_x = mask.x_min + x
                    tmp_image[tmp_y, tmp_x] = pixel[0:3]  # ignore alpha channel, len(channels) = 3 in normal image
        return tmp_image

    def save_image(self):
        cv2.imwrite("./data/images/masked_faces/" + self.filename, self.image)


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
        self.normal_image_paths = glob(self.normal_dir + "/*.jpg")  # list of every image
        self.normal_labels = pd.read_csv(self.labels_dir + "/normal_faces.csv")  # every label for every image used
        self.server = MJPEGServer()  # No GUI on Docker, output is over MJPEG server
        self.server.send_image(cv2.imread(self.normal_image_paths[0]))  # send initial image for testing
        server_thread = threading.Thread(target=self.server.start, args=(4000,))  # start server in another thread
        server_thread.start()

    def download_normal_faces(self):
        """
        Downloads 5000 images from OpenImages.
        """
        downloader = OpenImagesDownloader(5000)
        downloader.download_normal_faces()
        downloader.create_normal_csv()
        downloader.clean()
        self.normal_image_paths = glob(self.normal_dir + "/*.jpg")

    def mask(self):
        """
        Puts masks on every image
        """
        start = int(input("Image to start from (n): "))
        with open(self.labels_dir + "/masked_faces.csv", 'a') as mask_csv:
            if start == 0:
                mask_csv.write("key,label,xmin,ymin,xmax,ymax\n")
            for i, path in enumerate(self.normal_image_paths[start:]):
                print("Image #", start + i, ", percent done:",100*((start + i)/len(self.normal_image_paths)))
                normal_face = NormalFaceImage(path, self.normal_labels)
                labels = normal_face.get_labels()
                for _, label in labels.iterrows():
                    result = "d"  # try next mask
                    mask = Mask(label["xmin"], label["xmax"], label["ymin"], label["ymax"])
                    while result != "s" and mask != "i":  # while trying masks
                        tmp_image = normal_face.test_mask(mask)
                        self.server.send_image(tmp_image)
                        result = input("(s) Save mask, (a) ignore mask, (d) try next mask, (w) try another color: ")
                        if result == "s":  # saving this mask
                            out = "{},mask,{},{},{},{}\n".format(normal_face.filename, *mask.get_coordinates())
                            mask_csv.write(out)
                            normal_face.image = tmp_image
                        elif result == "a":  # ignoring this mask
                            break
                        elif result == "d":  # trying another orientation
                            mask.next_orientation()
                        elif result == "w":  # trying another color
                            mask.next_color()
                        else:  # unknown character
                            print("Invalid input.")
                normal_face.save_image()


if __name__ == "__main__":
    photoshop = Photoshop()
    photoshop.mask()
