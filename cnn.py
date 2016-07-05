#!/usr/bin/env python
"""
image encoder, using cnn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

import tensorflow as tf
from lm import RNN_SIZE
from coco_inputs import inputs

# images are in the range [-1, 1)
def encode_image(images, input_name = 'Mul'):
  PB_PATH = './data/models/graph.pb'

  graph_def = tf.GraphDef()
  graph_def.ParseFromString(open(PB_PATH).read())

  output_name = 'pool_3'
  cnn_dim = 2048

  pool_3_idx = [n.name for n in graph_def.node].index(output_name)
  valid_nodes = graph_def.node[:pool_3_idx+1]
  del graph_def.node[:]
  graph_def.node.extend(valid_nodes)

  name = 'cnn'
  graph = tf.get_default_graph()
  with graph.name_scope(name) as scope:
    tf.import_graph_def(graph_def, name=scope, input_map={input_name:images})
    output_node = graph.get_tensor_by_name(scope+output_name+':0')

  output_node = tf.reshape(output_node, [-1, cnn_dim])
  with tf.variable_scope('cnn2rnn'):
    w = tf.get_variable("proj_w", [cnn_dim, RNN_SIZE])
    b = tf.get_variable("proj_b", [RNN_SIZE])
    res = tf.matmul(output_node, w) + b

  return res

def main():
  import numpy as np
  data_dir = sys.argv[1]
  with tf.Graph().as_default():
    sess = tf.Session()
    images, _, image_paths = inputs(data_dir, True, 1)

    with tf.variable_scope("cnn"):
      img_enc0 = encode_image(images)

    with tf.variable_scope("cnn", reuse=True):
      image_path = image_paths[0]
      jpeg_content = tf.read_file(image_path)
      jpeg_enc = tf.image.decode_jpeg(jpeg_content)
      img_enc1 = encode_image(jpeg_content, "DecodeJpeg/contents")

    image = images[0, :, :, :]

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    while True:
      img_enc0_, img_enc1_, img_path_, image_ = sess.run([img_enc0, img_enc1, image_path, image])
      assert image_.max() > -1, "got empty image. check range of the distortion"
      print(img_path_)
      print(img_enc0_.shape, img_enc1_.shape)
      print(np.abs(img_enc0_ - img_enc1_).max(), img_enc0_.max(), img_enc0_.min())

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
