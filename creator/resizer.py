from typing import List
import cv2
import os
import numpy as np

class CSVFile:
    def __init__(self, filename):
        self.filename = filename
        self.fp = None # file pointer
        self.lines = []
        self.labels = []
        self.data = []

    def open(self) -> None:
        self.fp = open(self.filename, 'r')
        self.lines = self.fp.readlines()
        self.lines = [line.rstrip() for line in self.lines]
        self.labels = self.lines[0].split(',')
        self.data = [line.split(',') for line in self.lines[1:]]

    def get_unique_keys(self) -> List[str]:
        unique_keys = set()
        for line in self.data:
            unique_keys.add(line[0])
        ret = list(unique_keys)
        ret.sort()
        return ret

    def get_all_where_key_equals(self, key):
        return [(line[0], line[1], int(line[2]), int(line[3]), int(line[4]), int(line[5])) for line in self.data if line[0] == key]


class Resizer:
    def __init__(self, width: int, height: int):
        print("Make sure your working directory is in the creator folder")
        # Configurable
        self.width = width
        self.height = height
        # what to do with small images
        # "scale" means if we have 180x180 picture, it will be scaled up to 2000x2000 (if w & h are 2000)
        # "fit" means we will just keep it in the top left corner and add black bars
        # aspect ratio is always preserved
        self.mode = "scale" # "keep og"
        self.background = (0, 0, 0) # BGR not RGB
        # Technically Configurable but less commonly changed
        self.data_dir = "./data"
        self.labels_dir = self.data_dir + "/labels"
        self.images_dir = self.data_dir + "/images"
        self.normal_dir = self.images_dir + "/normal_faces"  # ground truth
        self.masks_dir = self.images_dir + "/masks" # pictures of masks that are overlayed
        self.masked_dir = self.images_dir + "/masked_faces" # edited photos with masks
        self.resized_truth_dir = self.images_dir + "/rs_truth" # resized ground truth
        self.resized_masked_dir = self.images_dir + "/rs_masked" # resized pics with edited masks
        self.resized_masks_dir = self.images_dir + "/rs_masks" # resized pics with the mask (what gets removed)

        if not os.path.exists(self.resized_truth_dir):
            os.makedirs(self.resized_truth_dir)
        if not os.path.exists(self.resized_masked_dir):
            os.makedirs(self.resized_masked_dir)
        if not os.path.exists(self.resized_masks_dir):
            os.makedirs(self.resized_masks_dir)

    def do_all(self):
        labels_masked_faces = CSVFile(self.labels_dir + '/masked_faces.csv')
        labels_masked_faces.open()
        unique_keys = labels_masked_faces.get_unique_keys()
        for u_key in unique_keys:
            print(u_key)
            normal_face_image_path = self.normal_dir + "/" + u_key
            masked_face_image_path = self.masked_dir + "/" + u_key
            resized_truth_path = self.resized_truth_dir + "/" + u_key
            resized_mask_channel_path = self.resized_masks_dir + "/" + u_key
            # IMREAD_UNCHANGED: preserve alpha channel
            normal_face_image = cv2.imread(normal_face_image_path, cv2.IMREAD_COLOR)
            masked_face_image = cv2.imread(masked_face_image_path, cv2.IMREAD_COLOR)
            if self.mode == "scale":
                dimensions = normal_face_image.shape
                og_height = dimensions[1]
                og_width = dimensions[0]
                # So we have to find the largest dimension
                image_orientation = "vertical" if og_height > og_width else "landscape"
                # now find how much to scale it by to fit in self.width or self.height
                # if self.height = 800, and our image is 600px, then
                # 800/600 = 1.33, so we multiple height & width by 1.33
                # if image is 1200px, then
                # 800/1200 = .67, so we multiple height & width by .67
                scaling_factor = self.height / og_height if image_orientation == "vertical" else self.width / og_width
                scaled_height = int(og_height * scaling_factor)
                scaled_width = int(og_width * scaling_factor)
                im_resized_truth = cv2.resize(normal_face_image, (scaled_height, scaled_width), interpolation=cv2.INTER_AREA)
                background = np.zeros((self.width, self.height, 3), np.uint8)
                background[:] = self.background
                y_offset = 0
                x_offset= 0
                background[y_offset:y_offset+im_resized_truth.shape[0], x_offset:x_offset+im_resized_truth.shape[1]] = im_resized_truth
                # output_truth = cv2.addWeighted(background, 1.0, im_resized_truth, 1.0, 0)
                cv2.imwrite(resized_truth_path, background)

                # create mask image
                mask_channel = np.zeros((og_width, og_height, 3), np.uint8)
                mask_channel[:] = (0, 0, 0)
                for mask in labels_masked_faces.get_all_where_key_equals(u_key):
                    (xmin, ymin, xmax, ymax) = (mask[2], mask[3], mask[4], mask[5])
                    # TODO: might need to do height - y because of opencv2's top-down logic
                    cv2.rectangle(mask_channel, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)
                cv2.imwrite(resized_mask_channel_path, mask_channel)

if __name__ == "__main__":
    resizer = Resizer(800, 800)
    resizer.do_all()
