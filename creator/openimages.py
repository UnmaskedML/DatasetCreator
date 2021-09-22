import json
import os
import sys
from glob import glob
from shutil import copyfile, rmtree
from zipfile import ZipFile

import pandas
from PIL import Image

from download import download_dataset


class OpenImagesDownloader:
    """
    Class for converting OpenImages format into an Axon usable format (TFRecords)
    """

    def __init__(self, nb_faces):
        # will error if not in Title Case
        self.labels = ["Human face"]
        self.title = "_".join([i.replace(" ", "") for i in self.labels])
        assert type(self.labels) == list
        self.limit = nb_faces
        assert type(self.limit) == int
        print("Getting dataset, size: {}, contents: {}".format(self.limit, self.labels))
        self.label_map = {}
        self.image_data = {}
        self.csv = []
        self.data_dir = "./data"

    def download(self):
        """
        API call to download OpenImages slice
        :return: None
        """
        download_dataset(dest_dir=self.data_dir + "/images", meta_dir="./data/meta", class_labels=self.labels,
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

    def create_csv(self):
        print("Parsing huge .csv. Give me a minute.")
        # get all jpgs
        images = glob(self.data_dir + "/images/*.jpg")
        with open(os.path.join("./data/meta/class-descriptions-boxable.csv")) as file:
            # save map of class ids to human-readable names
            for row in file.readlines():
                row = row.rstrip().split(',')
                self.label_map.update({row[0]: row[1]})

        try:
            # create image dir if it doesn't exist
            os.mkdir(self.data_dir + "/images")
        except FileExistsError:
            pass
        print("Copying...")
        for image_path in images:
            # save image size as only ratios are given
            im = Image.open(image_path)
            width, height = im.size
            file_id = image_path.split("/")[-1].rstrip(".jpg")
            self.image_data.update({file_id: {"height": height, "width": width}})
        print("Image metadata compiled.")

        # want to save only labels that are used.
        all_labels_df = pandas.read_csv(os.path.join("./data/meta/train-annotations-bbox.csv"))
        all_labels_df.set_index("ImageID", inplace=True)

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
        with open(self.data_dir + "/labels.csv", 'w+') as csv_file:
            csv_file.write("key,label,height,width,xmin,xmax,ymin,ymax\n")
            for line in self.csv:
                csv_file.write(",".join(map(str, line)) + "\n")

    def clean(self):
        rmtree(self.data_dir + "/meta")
        print("Cleaned", self.data_dir + "/meta")


if __name__ == "__main__":
    print("Python initialized")
    downloader = OpenImagesDownloader(5000)
    downloader.download()
    downloader.create_csv()
    downloader.clean()
    print("Complete")
