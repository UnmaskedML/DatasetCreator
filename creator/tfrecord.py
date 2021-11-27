import tensorflow as tf
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import io

import dataset_util


def create_tf_example(filename, rows):
    filename = filename.replace('.jpg', '_surgical.jpg')
    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = filename.encode('utf8')
    image_format = b'jpeg'

    xmins = [i["xmin"] for i in rows]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [i["xmax"] for i in rows]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [i["ymin"] for i in rows]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [i["ymax"] for i in rows]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [b"mask" for i in rows]  # List of string class name of bounding box (1 per box)
    classes = [1 for _ in rows]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(bytes(filename)),
        'image/source_id': dataset_util.bytes_feature(bytes(filename)),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def make_record(record_name, data):
    print(record_name)
    writer = tf.compat.v1.python_io.TFRecordWriter(f"data/tf/{record_name}.tfrecord")
    examples = []
    for image in data.keys():
        example = create_tf_example("data/images/annie_masked/" + image, data[image])
        examples.append(example)

    for example in examples:
        writer.write(example.SerializeToString())
    print("Wrote", len(examples), "examples")
    writer.close()
    print("Done")


def make_bboxes(data):
    masks = {}
    for img in data.keys():
        tmp_masks = []
        for mask in data[img]:
            # mask_xtl,mask_ytl,mask_xtm,mask_ytm,mask_xtr,mask_ytr,mask_xbr,mask_ybr,mask_xbm,mask_ybm,mask_xbl,mask_ybl
            cols = "mask_xtl,mask_ytl,mask_xtm,mask_ytm,mask_xtr,mask_ytr,mask_xbr,mask_ybr,mask_xbm,mask_ybm,mask_xbl,mask_ybl".split(
                ',')
            y_cols = [i for i in cols if 'y' in i]
            x_cols = [i for i in cols if 'x' in i]
            ymin = min([mask[i] for i in y_cols])
            ymax = max([mask[i] for i in y_cols])
            xmin = min([mask[i] for i in x_cols])
            xmax = max([mask[i] for i in x_cols])

            tmp_masks.append({"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "img": mask["img"]})
        masks.update({img: tmp_masks})
    return masks


if __name__ == '__main__':
    images = {}
    df = pd.read_csv("data/labels/annie_mask.csv")
    for index, row in df.iterrows():
        key = row["img"]
        try:
            images[key].append(row)
        except KeyError:
            images.update({key: [row]})
    train_X, test_X, train_y, test_y = train_test_split(list(images.keys()), list(images.values()), test_size=0.2)
    train = {i: images[i] for i in train_X}
    train = make_bboxes(train)
    test = {i: images[i] for i in test_X}
    test = make_bboxes(test)
    make_record("train", train)
    make_record("test", test)
