# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
import numpy as np

from PIL import Image

import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('original_uni_folder',
                                 './UM/UM2016/SegmentationClass',
                                 'Original ground truth annotations.')

tf.compat.v1.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.compat.v1.flags.DEFINE_string('output_dir',
                                 './UM/UM2016/SegmentationClassRaw',
                                 'folder to save modified ground truth annotations.')


def quantizetopalette(silf, palette, dither=False):
  """Convert an RGB or L mode image to use a given P image's palette."""
  silf.load()
  # use palette from reference image
  palette.load()
  if palette.mode != "P":
    raise ValueError("bad mode for palette image")
  if silf.mode != "RGB" and silf.mode != "L":
    raise ValueError(
      "only RGB or L mode images can be quantized to a palette"
    )
  im = silf.im.convert("P", 1 if dither else 0, palette.im)
  # the 0 above means turn OFF dithering
  # Later versions of Pillow (4.x) rename _makeself to _new
  try:
    return silf._new(im)
  except AttributeError:
    return silf._makeself(im)

def convert_to_P(filename):
  palettedata = []
  r = 0
  g = 0
  b = 0
  for i in range(0, 256):
    palettedata.extend([r, g, b])
    r += 1
    g = (g + 3) % 256
    b = (b + 7) % 256
  palimage = Image.new('P', (16, 16))
  palimage.putpalette(palettedata)
  oldimage = Image.open(filename)
  newimage = quantizetopalette(oldimage, palimage, dither=False)
  return newimage


# def _remove_colormap(file):
#   """Removes the color map from the annotation.

#   Args:
#     filename: Ground truth annotation filename.

#   Returns:
#     Annotation without color map.
#   """
#   arr_origin = np.array(file)
#   for i in range(0, arr_origin.shape[0]):
#     for j in range(0, arr_origin.shape[1]):
#       cur = arr_origin[i][j]
#       if cur   
#   return arr_origin


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.

  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.io.gfile.GFile(filename, mode='w') as f:
    pil_image.save(f, 'PNG')


def _remove_irrelevant(arr_origin):
    count = np.zeros((256,), dtype=int)
    for i in range(0, arr_origin.shape[0]):
        for j in range(0, arr_origin.shape[1]):
            count[arr_origin[i][j]] += 1
    
    for i in range(0, arr_origin.shape[0]):
        for j in range(0, arr_origin.shape[1]):
            if count[arr_origin[i][j]] < 500:
                if arr_origin[i][j] < 200:
                    arr_origin[i][j] = 0
                else:
                   arr_origin[i][j] = 255
    return arr_origin

def _change_channel(filename):
  img = Image.open(filename)
  l, r = img.size
  l = int(l / 5)
  r = int(r / 5)
  img = img.resize((l, r))
  arr_origin = np.array(img)
  arr_origin = arr_origin[:,:,0]
  arr_origin = _remove_irrelevant(arr_origin)
  #l, r = arr_origin.shape
#   l = int(l / 5)
#   r = int(r / 5)
#   np.resize(arr_origin,(l, r))
  return arr_origin

def main(unused_argv):
  # Create the output directory if not exists.
  if not tf.io.gfile.isdir(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  annotations = glob.glob(os.path.join(FLAGS.original_uni_folder,
                                       '*.' + FLAGS.segmentation_format))
  for annotation in annotations:
    raw_annotation = _change_channel(annotation)
    #raw_annotation = _remove_colormap(new_image)
    filename = os.path.basename(annotation)[:-4]
    _save_annotation(raw_annotation,
                     os.path.join(
                         FLAGS.output_dir,
                         filename + '.' + FLAGS.segmentation_format))


if __name__ == '__main__':
  tf.compat.v1.app.run()
