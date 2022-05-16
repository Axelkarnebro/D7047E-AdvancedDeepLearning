# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
"""MobileNet v3 models for Keras."""

from turtle import shape
import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import models,layers

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export




def MobileNetV3(stack_fn,
                input_shape,
                alpha=1.0,
                minimalistic=False):

  # Determine proper input shape and default size.
  # If both input_shape and input_tensor are used, they should match
  

  

  img_input = layers.Input(shape=input_shape)

  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
  if minimalistic:
      kernel = 3
      activation = relu
      se_ratio = None
  else:
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

  x = img_input
  x = layers.Conv2D(
      16,
      kernel_size=3,
      strides=(2, 2),
      padding='same',
      use_bias=False,
      name='Conv')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv/BatchNorm')(x)
  x = activation(x)

  x = stack_fn(x, kernel, activation, se_ratio)

  last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

  # if the width multiplier is greater than 1 we
  # increase the number of output channels
  if alpha > 1.0:
    last_point_ch = _depth(last_point_ch * alpha)
  x = layers.Conv2D(
      last_conv_ch,
      kernel_size=1,
      padding='same',
      use_bias=False,
      name='Conv_1')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv_1/BatchNorm')(x)
  x = activation(x)

  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  inputs = img_input

  # Create model.
  model = models.Model(inputs, x, name='MobilenetV3')
  return model


def MobileNetV3Small(alpha=1.0,input_shape = (150,150,1),minimalistic=False):
  def stack_fn(x, kernel, activation, se_ratio):

    def depth(d):
      return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
    x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
    x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation,
                            10)
    return x

  return MobileNetV3(stack_fn, input_shape, alpha,minimalistic)





def relu(x):
  return layers.ReLU()(x)


def hard_sigmoid(x):
  return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
  return layers.Multiply()([x, hard_sigmoid(x)])


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/
# slim/nets/mobilenet/mobilenet.py


def _depth(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v



def _se_block(inputs, filters, se_ratio, prefix):
  x = KDGlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(
          inputs)
  x = layers.Conv2D(
      _depth(filters * se_ratio),
      kernel_size=1,
      padding='same',
      name=prefix + 'squeeze_excite/Conv')(
          x)
  x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
  x = layers.Conv2D(
      filters,
      kernel_size=1,
      padding='same',
      name=prefix + 'squeeze_excite/Conv_1')(
          x)
  x = hard_sigmoid(x)
  x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
  return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
  shortcut = x
  prefix = 'expanded_conv/'
  infilters = backend.int_shape(x)[channel_axis]
  if block_id:
    # Expand
    prefix = 'expanded_conv_{}/'.format(block_id)
    x = layers.Conv2D(
        _depth(infilters * expansion),
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'expand')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand/BatchNorm')(
            x)
    x = activation(x)

  if stride == 2:
    x = layers.ZeroPadding2D(
        name=prefix + 'depthwise/pad')(
            x)
  x = layers.DepthwiseConv2D(
      kernel_size,
      strides=stride,
      padding='same' if stride == 1 else 'valid',
      use_bias=False,
      name=prefix + 'depthwise')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise/BatchNorm')(
          x)
  x = activation(x)

  if se_ratio:
    x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

  x = layers.Conv2D(
      filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      name=prefix + 'project')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project/BatchNorm')(
          x)

  if stride == 1 and infilters == filters:
    x = layers.Add(name=prefix + 'Add')([shortcut, x])
  return x


def MobileNetV3Large(alpha=1.0,input_shape = (150,150,1),minimalistic=False):

  def stack_fn(x, kernel, activation, se_ratio):

    def depth(d):
      return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
    x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
    x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
    x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
    x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation,
                            12)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation,
                            13)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation,
                            14)
    return x

  return MobileNetV3(stack_fn, input_shape, alpha,minimalistic)


class KDGlobalAveragePooling2D(layers.GlobalAveragePooling2D):

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.mean(inputs, axis=[1, 2], keepdims=True)
    else:
      return backend.mean(inputs, axis=[2, 3], keepdims=True)