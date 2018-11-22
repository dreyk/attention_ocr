# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Configuration to read FSNS dataset https://goo.gl/3Ldm8v."""

import os
import re
import tensorflow as tf
from tensorflow.contrib import slim
import logging

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data/fsns')

# The dataset configuration, should be used only as a default value.
DEFAULT_CONFIG = {
    'name': 'IDCAR',
    'splits': {
        'train': {
            'size': 10,
            'pattern': 'test/*.tfrecord'
        },
        'test': {
            'size': 2072,
            'pattern': 'test/*'
        },
        'validation': {
            'size': 2072,
            'pattern': 'test/*'
        }
    },
    'charset_filename': 'custom_charset.txt',
    'image_shape': (150, 150, 3),
    'num_of_views': 1,
    'max_sequence_length': 80,
    'null_code': 69,
    'items_to_descriptions': {
        'image': 'A [150 x 150 x 3] color image.',
        'label': 'Characters codes.',
        'text': 'A unicode string.',
        'length': 'A length of the encoded text.',
        'num_of_views': 'A number of different views stored within the image.'
    }
}


def read_charset(filename, null_character=u'\u2591'):
    """Reads a charset definition from a tab separated text file.

    charset file has to have format compatible with the FSNS dataset.

    Args:
      filename: a path to the charset file.
      null_character: a unicode character used to replace '<null>' character. the
        default value is a light shade block '░'.

    Returns:
      a dictionary with keys equal to character codes and values - unicode
      characters.
    """
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2).decode('utf-8')
            if char == '<nul>':
                char = null_character
            charset[code] = char
    return charset


class _NumOfViewsHandler(slim.tfexample_decoder.ItemHandler):
    """Convenience handler to determine number of views stored in an image."""

    def __init__(self, width_key, original_width_key, num_of_views):
        super(_NumOfViewsHandler, self).__init__([width_key, original_width_key])
        self._width_key = width_key
        self._original_width_key = original_width_key
        self._num_of_views = num_of_views

    def tensors_to_item(self, keys_to_tensors):
        return tf.to_int64(1)


class _NumOfViewsHandler(slim.tfexample_decoder.ItemHandler):
    """Convenience handler to determine number of views stored in an image."""

    def __init__(self, width_key, original_width_key, num_of_views):
        super(_NumOfViewsHandler, self).__init__([width_key, original_width_key])
        self._width_key = width_key
        self._original_width_key = original_width_key
        self._num_of_views = num_of_views

    def tensors_to_item(self, keys_to_tensors):
        return tf.to_int64(1)


class _CustomImage(slim.tfexample_decoder.ItemHandler):
    def __init__(self, image_key=None, shape=None,width_key='width',height_key='height'):
        if not image_key:
            image_key = 'image/encoded'
        super(_CustomImage, self).__init__([image_key,width_key,height_key])
        self._image_key = image_key
        self._width_key=width_key
        self._height_key=height_key
        self._w = float(shape[1])
        self._h = float(shape[0])

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        print(keys_to_tensors)
        image_buffer = keys_to_tensors[self._image_key]
        h = keys_to_tensors[self._height_key][0]
        w = keys_to_tensors[self._width_key][0]
        return self._decode(image_buffer,h,w)

    def _decode(self, image_buffer,original_h,original_w):
        image = tf.image.decode_png(image_buffer[0], channels=3)
        #print(original_h)
        #print(original_w)
        #image = tf.reshape(image,tf.TensorShape([original_h,original_w,3]))
        return self._resize(image,original_h,original_w)

    def _resize(self, im,original_h,original_w):
        w = tf.cast(original_w, tf.float32)
        h = tf.cast(original_h, tf.float32)
        ratio_w = tf.maximum(w / self._w, 1.0)
        ratio_h = tf.maximum(h / self._h, 1.0)
        ratio = tf.maximum(ratio_w, ratio_h)
        nw = tf.cast(w * ratio, tf.int32)
        nh = tf.cast(h * ratio, tf.int32)
        im = tf.image.resize_images(im, [nh, nw])
        padw = tf.maximum(0, int(self._w) - nw)
        padh = tf.maximum(0, int(self._h) - nh)
        im = tf.image.pad_to_bounding_box(im, 0, 0, nh + padh, nw + padw)
        return tf.reshape(im,[int(self._h),int(self._w),3])


def get_split(split_name, dataset_dir=None, config=None):
    """Returns a dataset tuple for FSNS dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources, by default it uses
        a predefined CNS path (see DEFAULT_DATASET_DIR).
      config: A dictionary with dataset configuration. If None - will use the
        DEFAULT_CONFIG.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR

    if not config:
        config = DEFAULT_CONFIG

    if split_name not in config['splits']:
        raise ValueError('split name %s was not recognized.' % split_name)

    logging.info('Using %s dataset split_name=%s dataset_dir=%s', config['name'],
                 split_name, dataset_dir)

    # Ignores the 'image/height' feature.
    zero = tf.zeros([1], dtype=tf.int64)
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature([1], tf.string, default_value=''),
        'width':
            tf.FixedLenFeature([1], tf.int64, default_value=zero),
        'height':
            tf.FixedLenFeature([1], tf.int64, default_value=zero),
        'orig_width':
            tf.FixedLenFeature([1], tf.int64, default_value=zero),
        'image/class':
            tf.FixedLenFeature([config['max_sequence_length']], tf.int64),
        'image/unpadded_class':
            tf.VarLenFeature(tf.int64),
        'image/text':
            tf.FixedLenFeature([1], tf.string, default_value=''),
    }
    items_to_handlers = {
        'image':
            _CustomImage(shape=config['image_shape'], image_key='image/encoded',width_key='width',height_key='height'),
        'label':
            slim.tfexample_decoder.Tensor(tensor_key='image/class'),
        'text':
            slim.tfexample_decoder.Tensor(tensor_key='image/text'),
        'num_of_views':
            _NumOfViewsHandler(
                width_key='width',
                original_width_key='orig_width',
                num_of_views=config['num_of_views'])
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)
    charset_file = os.path.join(dataset_dir, config['charset_filename'])
    charset = read_charset(charset_file)
    file_pattern = os.path.join(dataset_dir,
                                config['splits'][split_name]['pattern'])
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=config['splits'][split_name]['size'],
        items_to_descriptions=config['items_to_descriptions'],
        #  additional parameters for convenience.
        charset=charset,
        num_char_classes=len(charset),
        num_of_views=config['num_of_views'],
        max_sequence_length=config['max_sequence_length'],
        null_code=config['null_code'])
