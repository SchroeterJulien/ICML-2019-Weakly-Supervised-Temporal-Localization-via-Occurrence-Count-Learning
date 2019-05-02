# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create the recordio files necessary for training onsets and frames.
The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re

import librosa
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', 'Dataset',
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', './',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')

TEST_DIRS = ['ENSTDkCl/MUS', 'ENSTDkAm/MUS']
TRAIN_DIRS = ['AkPnBcht/MUS', 'AkPnBsdf/MUS', 'AkPnCGdD/MUS', 'AkPnStgb/MUS',
              'SptkBGAm/MUS', 'SptkBGCl/MUS', 'StbgTGd2/MUS']


def filename_to_id(filename):
  """Translate a .wav or .mid path to a MAPS sequence id."""
  return re.match(r'.*MUS-(.*)_[^_]+\.\w{3}',
                  os.path.basename(filename)).group(1)


def generate_train_set(exclude_ids):
  """Generate the train TFRecord."""
  train_file_pairs = []
  for directory in TRAIN_DIRS:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      if filename_to_id(wav_file) not in exclude_ids:
        train_file_pairs.append((wav_file, mid_file))

  return  [wav for wav, _ in train_file_pairs]

def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = []
  for directory in TEST_DIRS:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      test_file_pairs.append((wav_file, mid_file))

  return [filename_to_id(wav) for wav, _ in test_file_pairs], [wav for wav, _ in test_file_pairs]


def main():
  test_ids, test_samples = generate_test_set()
  train_samples = generate_train_set(test_ids)

