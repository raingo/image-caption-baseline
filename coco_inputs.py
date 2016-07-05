#!/usr/bin/env python
"""
read the tf record, supply the images and captions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import tensorflow as tf

from compile_data import MAX_SEQ_LEN
from gen_vocab import PAD, load_vocab, print_text

IM_S = 320
CNN_S = 299

def _parse_example_proto(example_serialized):
  # parse record
  # decode jpeg
  # random select one caption, convert it into integers
  # compute the length of the caption
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
      'image/coco-id': tf.FixedLenFeature([], dtype=tf.int64),
      'caption': tf.VarLenFeature(dtype=tf.string),
  }

  features = tf.parse_single_example(example_serialized, feature_map)

  cocoid = features['image/coco-id']
  image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
  # [0,255) --> [0,1)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  caption = tf.sparse_tensor_to_dense(features['caption'], default_value=".")
  caption = tf.random_shuffle(caption)[0]
  record_defaults = [[PAD]] * MAX_SEQ_LEN
  caption_tids = tf.decode_csv(caption, record_defaults)
  caption_tids = tf.pack(caption_tids)

  return image, caption_tids

def inputs(tf_dir, is_train, batch_size):
  image, caption_tids = records(tf_dir)

  reshaped_image = tf.image.resize_images(image, IM_S, IM_S)

  if is_train:
    distorted_image = tf.random_crop(reshaped_image, [CNN_S, CNN_S, 3])
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)
  else:
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, CNN_S, CNN_S)

  image = distorted_image

  # [0,1) --> [-1,1)
  image = tf.sub(image, 0.5)
  image = tf.mul(image, 2.0)

  num_preprocess_threads = 4
  min_queue_examples = 20

  outputs = [image, caption_tids]

  return tf.train.shuffle_batch(
      outputs,
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

def records(tf_dir):
  import glob
  files = glob.glob(osp.join(tf_dir, 'tf*'))
  filename_queue = tf.train.string_input_producer(files)

  reader = tf.TFRecordReader()
  _, example_serialized = reader.read(filename_queue)
  return _parse_example_proto(example_serialized)

def main():
  #test_func = records
  #test_func = lambda x: inputs(x, True, 10)
  test_func = lambda x: inputs(x, False, 10)
  _, i2w = load_vocab(sys.argv[2])
  with tf.Graph().as_default():
    sess = tf.Session()
    image, caption = test_func(sys.argv[1])

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    while True:
      outputs = sess.run([image, caption])
      print(outputs[1].shape, outputs[0].shape)
      print_text(outputs[1], i2w)

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
