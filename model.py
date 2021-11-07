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


def encoder(pl, num_fil, p_size, is_base=False):
    """
    Encoder function that applies downsampling over two previous convolutional layers
    :param pl: previous layer
    :param num_fil: number of filters in conv3d layer
    :param p_size: it will be used as pool size in downsampling
    :param is_base: default is False
    :return: layer, layer
    """
    # Conv block return value
    cbrv = _encoder(pl, num_fil)
    if is_base is True:
        return cbrv
    # Adding downsampling layer
    mp = MaxPooling3D(pool_size=p_size, strides=(2, 2, 2), padding="same")(cbrv)
    return cbrv, mp
