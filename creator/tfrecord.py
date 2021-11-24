import tensorflow as tf
import pandas as pd
from PIL import Image
import io

import dataset_util


def create_tf_example(filename, rows):
    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = filename.encode('utf8')
    image_format = b'jpeg'

    xmins = [min(i['mask_xtl'], i['mask_xbl'], i['mask_xtm'], i['mask_xbm']) for i in rows]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [max(i['mask_xbr'], i['mask_xtr']) for i in rows]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [min(i['mask_ytl'], i['mask_ytr'], i['mask_ytm']) for i in rows]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [max(i['mask_ybl'], i['mask_ybm'], i['mask_ybr']) for i in rows]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = ['mask'.encode("utf8") for i in rows]  # List of string class name of bounding box (1 per box)
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


def main():
    images = {}
    writer = tf.compat.v1.python_io.TFRecordWriter("data/tf/masks.tfrecord")
    df = pd.read_csv("../MaskTheFaceNew/mask.csv")
    for index, row in df.iterrows():
        key = row["img"]
        try:
            images[key].append(row)
        except KeyError:
            images.update({key: [row]})
    examples = []
    for image in images.keys():
        example = create_tf_example("../MaskTheFaceNew/normal_faces_masked/" + key.split('.')[0] + "_surgical.jpg", images[image])
        examples.append(example)

    for example in examples:
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    main()
