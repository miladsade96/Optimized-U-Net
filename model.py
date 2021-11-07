"""
    Optimized U-Net for Brain Tumor Segmentation Model Implementation
    Paper url: https://arxiv.org/pdf/2110.03352.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import (Input, Conv3D, Conv3DTranspose, Concatenate, LeakyReLU, MaxPooling3D)


def _encoder(pl, num_fil):
    """
    Encoder private function that only applies Conv3D, Instance Normalization and LeakyReLU
    :param pl: previous layer
    :param num_fil: number of filters in Conv3D layer
    :return: tensor
    """
    c_1 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(pl)
    in_1 = InstanceNormalization()(c_1)
    lr_1 = LeakyReLU(alpha=0.01)(in_1)  # Activation layer
    c_2 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(lr_1)
    in_2 = InstanceNormalization()(c_2)
    lr_2 = LeakyReLU(alpha=0.01)(in_2)  # Activation layer
    return lr_2
