# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""SSDFeatureExtractor for MobilenetV2 features."""

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2

slim = contrib_slim

from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim

from nets.mobilenet import conv_blocks
from nets.mobilenet import mobilenet as lib

op = lib.op

expand_input = conv_blocks.expand_input_by_factor

####based on local defs in slim/nets/mobilenet/mobilenet_v2.py
SMALL_V1 = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (conv_blocks.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(conv_blocks.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=24),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=24),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=64),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=2, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=320),
        #op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
    final_endpoint='layer_9',
    from_layer_names=['layer_5/expansion_output', 'layer_9', '', '', '', '']
)
# pyformat: enable

SMALL_X_V1 = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (conv_blocks.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(conv_blocks.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=24),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=24),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        #op(conv_blocks.expanded_conv, stride=2, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=2, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=320),
        #op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
    final_endpoint='layer_6',
    from_layer_names=['layer_3/expansion_output', 'layer_6', '', '', '', '']
)
# pyformat: enable

DEFAULT = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (conv_blocks.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(conv_blocks.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=24),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=24),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=64),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=160),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=320),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
    final_endpoint='layer_19',
    from_layer_names=['layer_15/expansion_output', 'layer_19', '', '', '', '']
)

SMALL_LK_V1 = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (conv_blocks.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(conv_blocks.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(conv_blocks.expanded_conv, stride=2, num_outputs=24, kernel_size = [3, 3]),
        op(conv_blocks.expanded_conv, stride=4, num_outputs=24),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=32),
        #op(conv_blocks.expanded_conv, stride=2, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=64),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=96),
        #op(conv_blocks.expanded_conv, stride=2, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=160),
        #op(conv_blocks.expanded_conv, stride=1, num_outputs=320),
        #op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
    final_endpoint='layer_6',
    from_layer_names=['layer_3/expansion_output', 'layer_6', '', '', '', '']
)

CONV_DEFS = {"default": DEFAULT, 
             "small_v1": SMALL_V1, 
             "small_x_v1":SMALL_X_V1,
             "small_lk_v1":SMALL_LK_V1}

class SSDCustomMobileNetV2FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV2 features."""

  def __init__(self,
               is_training,
               custom_conv_def,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False):
    """Customized version of MobileNetV2 Feature Extractor for SSD Models.

    Mobilenet v2 (experimental), designed by sandler@. More details can be found
    in //knowledge/cerebra/brain/compression/mobilenet/mobilenet_experimental.py

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDCustomMobileNetV2FeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        num_layers=num_layers,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
    self._custom_def = custom_conv_def

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    feature_map_layout = {
        'from_layer': CONV_DEFS[self._custom_def]["from_layer_names"][:self._num_layers],
        'layer_depth': [-1, -1, 512, 256, 256, 128][:self._num_layers],
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding,
    }

    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)), \
          slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=self._min_depth):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):

           #_, image_features = mobilenet_v2.mobilenet_base(
           #   ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
            #  final_endpoint='layer_19',
             # depth_multiplier=self._depth_multiplier,
              #use_explicit_padding=self._use_explicit_padding,
              #scope=scope)
           _, image_features =\
           mobilenet_v2.mobilenet(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                                  conv_defs = CONV_DEFS[self._custom_def],
                                  final_endpoint= CONV_DEFS[self._custom_def]["final_endpoint"],
                                  depth_multiplier=self._depth_multiplier,
                                  use_explicit_padding=self._use_explicit_padding,
                                  base_only = True,
                                  scope=scope)
           
        with slim.arg_scope(self._conv_hyperparams_fn()):
          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)

    return feature_maps.values()
