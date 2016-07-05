#!/usr/bin/env python
"""
convert tsv file of the format:
  image-id \t path \t caption
into tfrecords

python compile_data.py tsv-path vocab-path image-dir

encoded image. regular jpg file
caption in id
image-id as int

assume images are properly encoded jpg
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import os

from gen_vocab import load_vocab, tokenize, PAD, BOS, EOS, UNK
import random
import tensorflow as tf
import threading

NUM_PER_SHARDS = 2000
NUM_THREADS = 4
MAX_SEQ_LEN = 100

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

import imghdr
from cStringIO import StringIO
from PIL import Image

def _ensure_jpeg(path):
  image_buffer = tf.gfile.FastGFile(path, 'r')

  if imghdr.what(image_buffer) != 'jpeg':
    img = Image.open(image_buffer)
    buffer = StringIO()
    img.save(buffer, format='jpeg')
    res = buffer.getvalue()
    print("converted non jpeg:", path)
  else:
    res = image_buffer.read()
  return res

def _convert_to_example(pid, path, ids):

  ids_str = []
  for tokens in ids:
    tokens = [BOS] + tokens + [EOS] + [PAD]*MAX_SEQ_LEN
    tokens = tokens[:MAX_SEQ_LEN]
    tokens = [str(i) for i in tokens]
    ids_str.append(','.join([str(i) for i in tokens]))

  image_buffer = _ensure_jpeg(path)
  example = tf.train.Example(
      features=tf.train.Features(feature={
      'image/coco-id': _int64_feature(pid),
      'image/path': _bytes_feature(path),
      'caption': _bytes_feature(ids_str),
      'image/encoded': _bytes_feature(image_buffer)}))

  return example

def _process_threads(tid, num_threads, data, save_dir, w2i, name='tf'):
  cnt = 0
  for idx in range(tid, len(data), num_threads):
    if cnt % NUM_PER_SHARDS == 0:
      output_file = osp.join(save_dir, '%s-t%02d-s%05d' % (name, tid, cnt/NUM_PER_SHARDS))
      writer = tf.python_io.TFRecordWriter(output_file)

      if tid == 0:
        print(cnt, tid, idx, len(data))

    fields = data[idx]
    image_id = int(fields[0])
    image_path = fields[1]
    ids = []
    for f in fields[2]:
      ids.append([w2i[w] if w in w2i else UNK for w in tokenize(f)])

    example = _convert_to_example(image_id, image_path, ids)
    writer.write(example.SerializeToString())
    cnt += 1

def main():
  tsv_path = sys.argv[1]
  vocab_path = sys.argv[2]
  image_dir = sys.argv[3]

  valid_path = osp.join(osp.dirname(tsv_path), 'checks.valid')

  valid = set()
  if osp.exists(valid_path):
    with open(valid_path) as reader:
      for line in reader:
        valid.add(line.strip())

  print("#valid:", len(valid))

  save_dir = tsv_path + '.tf'
  import shutil
  if osp.exists(save_dir):
    shutil.rmtree(save_dir)
  os.mkdir(save_dir)

  w2i, _ = load_vocab(vocab_path)

  from collections import defaultdict
  data = defaultdict(list)
  path2id = {}

  with open(tsv_path) as reader:
    for line in reader:
      fields = line.strip().split('\t')
      name = osp.basename(fields[1])
      if len(valid) > 0 and name not in valid:
        print("Hit")
        continue

      fields[1] = osp.join(image_dir, fields[1])
      if not osp.exists(fields[1]):
        continue
      data[(fields[0], fields[1])].append(fields[2])
      path2id[fields[1]] = fields[0]

  data_ = []
  for key, captions in data.items():
    assert path2id[key[1]] == key[0], "path and image-id is not one2one"
    data_.append(key + (captions,))
  random.shuffle(data_)

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  threads = []
  for thread_index in range(NUM_THREADS):
    args = (thread_index, NUM_THREADS, data_, save_dir, w2i)
    t = threading.Thread(target=_process_threads, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
