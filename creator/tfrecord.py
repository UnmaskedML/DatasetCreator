import tensorflow as tf
import pandas as pd
from PIL import Image
import io

import dataset_util


def create_tf_example(filename):
    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = filename.encode('utf8')
    image_format = b'jpeg'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg_io),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    examples = []
    writer = tf.compat.v1.python_io.TFRecordWriter("data")
    with pd.read_csv("data/labels/masked_faces.csv") as df:
        for index, row in df.iterrows():
            key = row["key"]
            example = create_tf_example("data/images/masked_faces/" + key)
            examples.append(example)

    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
