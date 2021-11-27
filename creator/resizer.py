from typing import List
import cv2
import os
import numpy as np
import random

class MaskCSVRow:
    def __init__(self, img, img_height, img_width, face_xtl, face_ytl, face_xbr, face_ybr,
                 mask_xtl, mask_ytl, mask_xtm, mask_ytm, mask_xtr, mask_ytr, mask_xbr,
                 mask_ybr,mask_xbm,mask_ybm,mask_xbl,mask_ybl):
        self.img = img
        self.img_height = img_height
        self.img_width = img_width
        self.face_xtl = face_xtl
        self.face_ytl = face_ytl
        self.face_xbr = face_xbr
        self.face_ybr = face_ybr
        self.mask_xtl = mask_xtl
        self.mask_ytl = mask_ytl
        self.mask_xtm = mask_xtm
        self.mask_ytm = mask_ytm
        self.mask_xtr = mask_xtr
        self.mask_ytr = mask_ytr
        self.mask_xbr = mask_xbr
        self.mask_ybr = mask_ybr
        self.mask_xbm = mask_xbm
        self.mask_ybm = mask_ybm
        self.mask_xbl = mask_xbl
        self.mask_ybl = mask_ybl

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
        return [MaskCSVRow(
            line[0], # img
            int(line[1]), # img_height
            int(line[2]), # img_width
            int(line[3]), # face_xtl
            int(line[4]), # face_ytl
            int(line[5]), # face_xbr
            int(line[6]), # face_ybr
            int(line[7]), # mask_xtl
            int(line[8]), # mask_ytl
            int(line[9]), # mask_xtm
            int(line[10]), # mask_ytm
            int(line[11]), # mask_xtr
            int(line[12]), # mask_ytr
            int(line[13]), # mask_xbr
            int(line[14]), # mask_ybr
            int(line[15]), # mask_xbm
            int(line[16]), # mask_ybm
            int(line[17]), # mask_xbl
            int(line[18]) # mask_ybl
                ) for line in self.data if line[0] == key]


def is_for_training():
    return random.randrange(1, 80) > 20

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
        self.mode = "scale"
        self.background = (0, 0, 0) # allegedly BGR not RGB
        # Technically Configurable but less commonly changed
        self.output_csv = "./data/images/rs_data.csv"
        self.mask_labels = "../MaskTheFaceNew/mask.csv"
        self.normal_dir = "../MaskTheFaceNew/normal_faces"  # ground truth
        self.masked_dir = "../MaskTheFaceNew/normal_faces_masked" # edited photos with masks
        self.resized_truth_training_dir =  "./data/images/rs_truth_training" # resized ground truth
        self.resized_truth_test_dir =      "./data/images/rs_truth_test" # resized ground truth
        self.resized_masks_training_dir =  "./data/images/rs_masks_training" # resized pics with the mask (what gets removed)
        self.resized_masks_test_dir =      "./data/images/rs_masks_test" # resized pics with the mask (what gets removed)
        self.resized_masked_training_dir = "./data/images/rs_masked_training" # resized pics with edited masks
        self.resized_masked_test_dir =     "./data/images/rs_masked_test" # resized pics with edited masks

        for dir in [self.resized_truth_training_dir, self.resized_truth_test_dir, self.resized_masks_training_dir,
                    self.resized_masks_test_dir, self.resized_masked_training_dir, self.resized_masked_test_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)


    def do_all(self):
        labels_masked_faces = CSVFile(self.mask_labels)
        labels_masked_faces.open()
        unique_keys = labels_masked_faces.get_unique_keys()

        csv_output = open(self.output_csv, 'w')
        csv_output.write("image_id,mask_num,xmin,ymin,xmax,ymax\n")

        for u_key in unique_keys:
            # this image is first when alphanumerically sorted in windows
            #if u_key != "00a0d634ad200ced.jpg":
            #    continue
            print(u_key)
            image_id = u_key.split('.')[0]
            # What we read
            normal_face_image_path = self.normal_dir + "/" + u_key
            masked_face_image_path = self.masked_dir + "/" + image_id + '_surgical.jpg'
            # What we'll write
            resized_truth_training_path = self.resized_truth_training_dir + "/" + u_key
            resized_truth_test_path = self.resized_truth_test_dir + "/" + u_key
            resized_masked_training_path = self.resized_masked_training_dir + "/" + image_id + '_surgical.jpg'
            resized_masked_test_path = self.resized_masked_test_dir + "/" + image_id + '_surgical.jpg'

            def gen_mask_channel_path(is_training, num):
                """
                num: some images have multiple masks, so there is 1 mask channel image per mask per image
                """
                if is_training:
                    return f"{self.resized_masks_training_dir}/{image_id}_{num}.jpg"
                else:
                    return f"{self.resized_masks_test_dir}/{image_id}_{num}.jpg"
            # Load pictures into memory
            normal_face_image = cv2.imread(normal_face_image_path, cv2.IMREAD_COLOR)
            masked_face_image = cv2.imread(masked_face_image_path, cv2.IMREAD_COLOR)
            if self.mode == "scale":
                is_training_image = is_for_training()
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
                # resize image
                im_resized_truth = cv2.resize(normal_face_image, (scaled_height, scaled_width), interpolation=cv2.INTER_AREA)
                # create the background
                background = np.zeros((self.width, self.height, 3), np.uint8)
                background[:] = self.background
                # paste image onto background (black bars for aspect ratio)
                y_offset = 0
                x_offset= 0
                background[y_offset:y_offset+im_resized_truth.shape[0], x_offset:x_offset+im_resized_truth.shape[1]] = im_resized_truth
                # output_truth = cv2.addWeighted(background, 1.0, im_resized_truth, 1.0, 0)
                cv2.imwrite(resized_truth_training_path if is_training_image else resized_truth_test_path, background)

                # create mask channel image
                mask_bounding_boxes = labels_masked_faces.get_all_where_key_equals(u_key)
                mask_num = 1
                for mask in mask_bounding_boxes:
                    new_mask_channel = np.zeros((og_width, og_height, 3), np.uint8)
                    new_mask_channel[:] = (0, 0, 0)
                    minx = min(mask.mask_xtl, mask.mask_xbl, mask.mask_xtm, mask.mask_xbm)
                    miny = min(mask.mask_ytl, mask.mask_ytr, mask.mask_ytm)
                    maxx = max(mask.mask_xbr, mask.mask_xtr)
                    maxy = max(mask.mask_ybl, mask.mask_ybm, mask.mask_ybr)
                    # opencv2's top-down logic
                    cv2.rectangle(new_mask_channel,
                                  (minx, miny),
                                  (maxx, maxy),
                                  (255, 255, 255), -1)
                    # resize image
                    new_mask_channel = cv2.resize(new_mask_channel, (scaled_height, scaled_width), interpolation=cv2.INTER_AREA)
                    # create the background
                    background = np.zeros((self.width, self.height, 3), np.uint8)
                    background[:] = self.background
                    # paste image onto background (black bars for aspect ratio)
                    # yes we're pasting a mostly black image onto black
                    y_offset = 0
                    x_offset= 0
                    background[y_offset:y_offset+new_mask_channel.shape[0], x_offset:x_offset+new_mask_channel.shape[1]] = new_mask_channel
                    path = gen_mask_channel_path(is_training_image, mask_num)
                    cv2.imwrite(path, background)
                    csv_output.write(f"{image_id},{mask_num},{minx},{miny},{maxx},{maxy}\n")
                    mask_num += 1

                # resize image
                masked_face_image = cv2.resize(masked_face_image, (scaled_height, scaled_width), interpolation=cv2.INTER_AREA)
                # create the background
                background = np.zeros((self.width, self.height, 3), np.uint8)
                background[:] = self.background
                # paste image onto background (black bars for aspect ratio)
                y_offset = 0
                x_offset= 0
                background[y_offset:y_offset+masked_face_image.shape[0], x_offset:x_offset+masked_face_image.shape[1]] = masked_face_image
                cv2.imwrite(resized_masked_training_path if is_training_image else resized_masked_test_path, background)

        csv_output.close()

if __name__ == "__main__":
    resizer = Resizer(800, 800)
    resizer.do_all()
