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
import simplejson

def dump_json(split):
  save_path = split + '.tsv'
  json_path = osp.join('./annotations/', 'captions_%s.json' % split)
  image_only = False
  if not osp.exists(json_path):
    json_path = osp.join('./annotations/', 'image_info_%s.json' % split)
    image_only = True

  with open(json_path) as reader:
    info = simplejson.load(reader)

  images = info['images']
  with open(save_path, 'w') as writer:
    # image-id image-path caption
    if image_only:
      for image in images:
        print(image['id'],
            osp.join(split, image['file_name']),
            ".", sep='\t', file=writer)
    else:
      id2path = {item['id']:osp.join(split, item['file_name'])
          for item in images}
      for ann in info['annotations']:
        for caption in ann['caption'].split('\n'):
          caption = caption.strip()
          if len(caption) > 0:
            print(ann['image_id'],
                id2path[ann['image_id']],
                caption,
                sep='\t', file=writer)

def main():
  dump_json('train2014')
  dump_json('val2014')
  dump_json('test2015')
  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
