#!/usr/bin/env python
"""
language model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

import tensorflow as tf
from gen_vocab import PAD, load_vocab, BOS, EOS
from compile_data import MAX_SEQ_LEN
from coco_inputs import inputs

RNN_SIZE = 500
LEARNING_RATE = 1e-2
MAX_GRADIENT_NORM = 5.

def _seq_len(data):
  mask = tf.not_equal(data, PAD)
  mask = tf.cast(mask, tf.int32)
  return tf.reduce_sum(mask, 1), mask

def build_lm(text_inputs, num_symbols, cond, seq_len=None):
  if seq_len is None:
    seq_len = _seq_len(text_inputs)[0]

  cell = tf.nn.rnn_cell.GRUCell(RNN_SIZE)

  with tf.variable_scope("rnn_inputs"):
    embedding = tf.get_variable("embedding",
            [num_symbols, RNN_SIZE])
    rnn_inputs = tf.nn.embedding_lookup(embedding, text_inputs)

  with tf.variable_scope("decoder"):
    if seq_len == 1:
      with tf.variable_scope("RNN"):
        # hacky. RNN comes from the rnn.py code
        rnn_inputs = tf.reshape(rnn_inputs, [-1, RNN_SIZE])
        output, state = cell(rnn_inputs, cond)
    else:
      output, state = tf.nn.dynamic_rnn(cell, rnn_inputs,
          initial_state = cond,
          dtype=tf.float32,
          sequence_length=seq_len)

  with tf.variable_scope("outputs"):
    w = tf.get_variable("proj_w", [RNN_SIZE, num_symbols])
    b = tf.get_variable("proj_b", [num_symbols])

    output = tf.reshape(output, [-1, RNN_SIZE])
    text_outputs = tf.matmul(output, w)
    text_outputs += b

  return text_outputs, state

def lm_loss(text, num_symbols, cond):
  org_shape = tf.shape(text)
  batch_size = org_shape[0]

  text_inputs = tf.slice(text, [0,0], [-1,MAX_SEQ_LEN-1])
  text_targets = tf.slice(text, [0,1], [-1,-1])
  seq_len, mask = _seq_len(text_targets)

  logits, _ = build_lm(text_inputs, num_symbols, cond)

  text_targets = tf.reshape(text_targets, [-1])
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, text_targets)
  xent = tf.reshape(xent, [batch_size, -1])

  mask = tf.cast(mask, tf.float32)
  seq_len = tf.cast(seq_len, tf.float32)

  xent_masked = xent * mask
  xent_time_axis = tf.reduce_sum(xent_masked, 1) / seq_len
  return tf.reduce_mean(xent_time_axis)

def build_sampler(num_symbols, cond, batch_size):
  text = tf.constant(BOS, shape=[batch_size, 1])

  texts = []
  for i in range(MAX_SEQ_LEN):
    logits, cond = build_lm(text, num_symbols, cond)
    text = tf.argmax(logits, 1)
    texts.append(text)
    text = tf.expand_dims(text, 1)

  # return batch_size * MAX_SEQ_LEN
  return tf.transpose(tf.pack(texts))

def main():
  data_dir = sys.argv[1]
  vocab_path = sys.argv[2]
  _, i2w = load_vocab(vocab_path)
  num_symbols = len(i2w)
  print('num_symbols:', num_symbols)

  with tf.Graph().as_default():
    sess = tf.Session()
    captions = inputs(data_dir, True, 10)[1]
    with tf.variable_scope("lm"):
      loss = lm_loss(captions, num_symbols, None)

    with tf.variable_scope("lm", reuse=True):
      batch_size = 10
      cond = tf.random_normal([batch_size, RNN_SIZE])
      samples = build_sampler(num_symbols, cond, batch_size)

    params = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(LEARNING_RATE)

    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     MAX_GRADIENT_NORM)

    global_step = tf.Variable(0, trainable=False)
    train_op = opt.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10000):
      if i % 100 == 0:
        samples_ = sess.run([samples])[0]
        print("samples at iteration", i)
        for sample in samples_:
          tokens = []
          for ii in sample:
            if ii == EOS:
              break
            tokens.append(i2w[ii])
          print(" ", ' '.join(tokens))

      print(i, sess.run([train_op, loss])[1])

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
