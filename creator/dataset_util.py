import tensorflow as tf

def int64_feature(value):
  return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=value))


def bytes_feature(value):
  return tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=value))


def float_feature(value):
  return tf.compat.v1.train.Feature(float_list=tf.compat.v1.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.compat.v1.train.Feature(float_list=tf.compat.v1.train.FloatList(value=value))


