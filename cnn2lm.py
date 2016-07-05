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
from gen_vocab import load_vocab, EOS, print_text
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

  do_train = True
  try:
    eval_save_path = sys.argv[4]
    do_train = eval_save_path == 'train'
  except:
    pass

  batch_size = 32

  _, i2w = load_vocab(vocab_path)
  num_symbols = len(i2w)
  print('num_symbols:', num_symbols)
  with tf.Graph().as_default():
    sess = tf.Session()
    with tf.device('/cpu:0'):
      images, captions, coco_ids = inputs(data_dir,
          do_train,
          batch_size,
          None if do_train else 2)

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

    init_op = tf.group(tf.initialize_all_variables(),
                  tf.initialize_local_variables())

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tf.get_default_graph().finalize()

    if osp.exists(ckpt_path):
      saver.restore(sess, ckpt_path)

    start = global_step.eval(session=sess)

    def _eval(output=sys.stdout):
      samples_, paths_ = sess.run([samples, coco_ids])
      print_text(samples_, i2w, paths_, file=output)

    def train():
      max_iters = 8000 * 20

      for i in range(start, max_iters):
        if i % 1000 == 0:
          saver.save(sess, ckpt_path, write_meta_graph=False)
          print("samples at iteration", i)
          _eval()

        _loss = sess.run([train_op, loss])[1]

        if i % 100 == 0:
          print(i, max_iters, _loss, strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    def eval():
      try:
        with open(eval_save_path + '-%d' % start, 'w') as writer:
          cnt = 0
          while not coord.should_stop():
            _eval(writer)
            cnt += 1
            print(cnt)
      except tf.errors.OutOfRangeError:
        print('finish eval')

    if do_train:
      train()
    else:
      eval()

    #coord.request_stop()
    #coord.join(threads)
    #sess.close()


if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
