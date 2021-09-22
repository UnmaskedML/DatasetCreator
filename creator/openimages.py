import os
from glob import glob
from shutil import rmtree

import pandas
from PIL import Image

from download import download_dataset


class OpenImagesDownloader:
    """
    Class for converting OpenImages format into an Axon usable format (TFRecords)
    """

    def __init__(self, nb_faces: int):
        # will error if not in Title Case
        self.labels = ["Human face"]
        self.limit = nb_faces
        print("Getting dataset, size: {}, contents: {}".format(self.limit, self.labels))
        self.label_map = {}
        self.image_data = {}
        self.csv = []
        self.data_dir = "./data"
        self.labels_dir = self.data_dir + "/labels"
        self.meta_dir = self.data_dir + "/meta"  # class descriptions and all labels go here
        self.images_dir = self.data_dir + "/images"
        self.normal_dir = self.images_dir + "/normal_faces"  # unmasked faces
        self.masks_dir = self.images_dir + "/masks"
        self.mask_dir = self.images_dir + "/masked_faces"

    def download_normal_faces(self):
        """
        API call to download OpenImages slice
        :return: None
        """
        download_dataset(dest_dir=self.normal_dir, meta_dir=self.meta_dir, class_labels=self.labels,
                         exclusions_path=None, limit=self.limit)

    def parse_line(self, key, label, height, width, box):
        """
        Add a line to the csv
        :param key: key of image
        :param label: specific label for this line
        :param height: height of image
        :param width: width of image
        :param box: ratio bounding box for label
        :return: None
        """
        self.csv.append(
            [key, label, height, width, int(box["xmin"] * width), int(box["xmax"] * width), int(box["ymin"] * height),
             int(box["ymax"] * height)])

    def create_normal_csv(self):
        print("Parsing huge .csv. Give me a minute.")
        # get all jpgs
        images = glob(self.normal_dir + "/*.jpg")
        with open(self.meta_dir + "/class-descriptions-boxable.csv") as file:
            # save map of class ids to human-readable names
            for row in file.readlines():
                row = row.rstrip().split(',')
                self.label_map.update({row[0]: row[1]})

        try:
            # create image dir if it doesn't exist
            os.mkdir(self.images_dir)
        except FileExistsError:
            pass
        print("Gathering image metadata")
        for image_path in images:
            # save image size as only ratios are given
            im = Image.open(image_path)
            width, height = im.size
            file_id = image_path.split("/")[-1].rstrip(".jpg")
            self.image_data.update({file_id: {"height": height, "width": width}})

        # want to save only labels that are used.
        all_labels_df = pandas.read_csv(self.meta_dir + "/train-annotations-bbox.csv")
        all_labels_df.set_index("ImageID", inplace=True)
        print("Collecting labels")
        labels = [i.lower() for i in self.labels]
        for key in self.image_data.keys():
            entry = all_labels_df.loc[key]
            try:
                height = self.image_data[key]["height"]
                width = self.image_data[key]["width"]
                # multiple labels for this image
                if type(entry) == pandas.DataFrame:
                    for row in entry.iterrows():
                        box = {i.lower(): row[1][i] for i in ["XMin", "XMax", "YMin", "YMax"]}
                        label = self.label_map[row[1]["LabelName"]].lower()
                        if label in labels:
                            self.parse_line(key + '.jpg', label, height, width, box)
                else:
                    # one label for this image
                    box = {i.lower(): entry.get(i) for i in ["XMin", "XMax", "YMin", "YMax"]}
                    label = self.label_map[entry.get("LabelName")].lower()
                    if label in labels:
                        self.parse_line(key + '.jpg', label, height, width, box)
            except KeyError:
                # no labels for this image. why did we download it?
                pass

        print("Writing label file")
        try:
            # create image dir if it doesn't exist
            os.mkdir(self.labels_dir)
        except FileExistsError:
            pass
        with open(self.labels_dir + "/normal_faces.csv", 'w+') as csv_file:
            csv_file.write("key,label,height,width,xmin,xmax,ymin,ymax\n")
            for line in self.csv:
                csv_file.write(",".join(map(str, line)) + "\n")

    def clean(self):
        rmtree(self.meta_dir)
        print("Cleaned", self.meta_dir)
