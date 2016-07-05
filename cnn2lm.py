#!/usr/bin/env python
"""
Code description
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import tensorflow as tf
from lm import lm_loss, build_sampler, MAX_GRADIENT_NORM
from cnn import encode_image
from gen_vocab import load_vocab, EOS
from coco_inputs import inputs
from time import gmtime, strftime

LEARNING_RATE = 1e-3

def image2text(images, captions, num_symbols):
  cnn = encode_image(images)
  loss = lm_loss(captions, num_symbols, cnn)
  return loss

def main():
  data_dir = sys.argv[1]
  vocab_path = sys.argv[2]
  ckpt_path = sys.argv[3]

  batch_size = 32

  _, i2w = load_vocab(vocab_path)
  num_symbols = len(i2w)
  print('num_symbols:', num_symbols)
  with tf.Graph().as_default():
    sess = tf.Session()
    images, captions = inputs(data_dir, True, batch_size)

    with tf.variable_scope("im2txt"):
      loss = image2text(images, captions, num_symbols)

    with tf.variable_scope("im2txt", reuse=True):
      cnn = encode_image(images)
      samples = build_sampler(num_symbols, cnn, batch_size)

    params = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(LEARNING_RATE)

    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     MAX_GRADIENT_NORM)

    global_step = tf.Variable(0, trainable=False)
    train_op = opt.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    max_iters = 8000 * 20

    for i in range(max_iters):
      if i % 1000 == 0:
        saver.save(sess, ckpt_path)
        samples_ = sess.run([samples])[0]
        print("samples at iteration", i)
        for sample in samples_:
          tokens = []
          for ii in sample:
            if ii == EOS:
              break
            tokens.append(i2w[ii])
          print(" ", ' '.join(tokens))

      _loss = sess.run([train_op, loss])[1]

      if i % 100 == 0:
        print(i, max_iters, _loss, strftime("%Y-%m-%d %H:%M:%S", gmtime()))

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
