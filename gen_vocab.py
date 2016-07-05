#!/usr/bin/env python
"""
build vocabulary

cat *.tsv | python gen_vocab.py save-dir

output vocab in save-dir, sorted by frequency
token frequency

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

from nltk.tokenize import word_tokenize

SP_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
PAD = 0
BOS = 1
EOS = 2
UNK = 3

def tokenize(sent):
  tokens = word_tokenize(sent.lower())
  if tokens[-1] == '.':
    tokens.pop()
  return tokens

def load_vocab(vocab_path):
  i2w = []
  with open(vocab_path) as reader:
    for line in reader:
      i2w.append(line.split()[0])
  w2i= {w:idx for idx, w in enumerate(i2w)}
  return w2i, i2w

def main():
  save_dir = sys.argv[1]

  from collections import Counter

  vocab = Counter()
  for line in sys.stdin:
    fields = line.strip().split('\t')
    if len(fields) < 3:
      print(line.strip())
      continue
    vocab.update(tokenize(fields[2]))
  with open(osp.join(save_dir, 'vocab'), 'w') as writer:
    for token in SP_TOKENS:
      print(token, 100000000, file=writer)
    for w, n in vocab.most_common():
      print(w, n, file=writer)
  pass


if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
