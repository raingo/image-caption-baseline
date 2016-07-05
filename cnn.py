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
def encode_image(images):
  PB_PATH = './data/models/graph.pb'

  graph_def = tf.GraphDef()
  graph_def.ParseFromString(open(PB_PATH).read())

  input_name = 'Mul'
  output_name = 'pool_3'
  cnn_dim = 2048

  pool_3_idx = [n.name for n in graph_def.node].index(output_name)
  valid_nodes = graph_def.node[:pool_3_idx+1]
  del graph_def.node[:]
  graph_def.node.extend(valid_nodes)

  name = 'cnn'
  tf.import_graph_def(graph_def, name = name, input_map={input_name:images})
  graph = tf.get_default_graph()
  with tf.variable_scope("cnn") as vs:
    output_node = graph.get_tensor_by_name(vs.name + '/' + output_name+':0')

  with tf.variable_scope('cnn2rnn'):
    output_node = tf.reshape(output_node, [-1, cnn_dim])
    w = tf.get_variable("proj_w", [cnn_dim, RNN_SIZE])
    b = tf.get_variable("proj_b", [RNN_SIZE])
    res = tf.matmul(output_node, w) + b

  return res

def main():
  data_dir = sys.argv[1]
  with tf.Graph().as_default():
    sess = tf.Session()
    images, _ = inputs(data_dir, True, 10)
    img_enc = encode_image(images)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    while True:
      print(sess.run(img_enc).shape)

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
