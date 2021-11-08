"""
    Optimized U-Net for Brain Tumor Segmentation Model Implementation
    Paper url: https://arxiv.org/pdf/2110.03352.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, Concatenate, LeakyReLU, MaxPooling3D, Input)


def _encoder(pl, num_fil, is_first_conv=False):
    """
    Encoder private function that only applies Conv3D, Instance Normalization and LeakyReLU
    :param pl: previous layer
    :param num_fil: number of filters in Conv3D layer
    :param is_first_conv: whether this conv block is the first one or not
    :return: tensor
    """
    fcls = (2, 2, 2)    # first convolutional layer strides
    scls = (1, 1, 1)    # second convolutional layer strides
    # first conv block condition
    if is_first_conv:
        fcls = (1, 1, 1)

    c_1 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=fcls, padding="same")(pl)
    in_1 = InstanceNormalization()(c_1)
    lr_1 = LeakyReLU(alpha=0.01)(in_1)  # Activation layer
    c_2 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=scls, padding="same")(lr_1)
    in_2 = InstanceNormalization()(c_2)
    lr_2 = LeakyReLU(alpha=0.01)(in_2)  # Activation layer
    return lr_2


def encoder(pl, num_fil, is_first_conv=False, is_base=False):
    """
    Encoder function that applies downsampling over two previous convolutional layers
    :param pl: previous layer
    :param num_fil: number of filters in conv3d layer
    :param is_first_conv: whether this conv block is the first one or not
    :param is_base: default is False
    :return: tensor, tensor
    """
    # Conv block return value
    cbrv = _encoder(pl, num_fil, is_first_conv)
    if is_base is True:
        return cbrv
    # Adding downsampling layer
    mp = MaxPooling3D(strides=(2, 2, 2), padding="same")(cbrv)
    return cbrv, mp


def _decoder(pl, num_fil):
    """
    Private decoder function that only applies Conv3D, Instance Normalization and LeakyReLU
    :param pl: previous layer
    :param num_fil: number of filters in conv3d layer
    :return: tensor
    """
    c_1 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(pl)
    in_1 = InstanceNormalization()(c_1)
    lr_1 = LeakyReLU(alpha=0.01)(in_1)
    c_2 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(lr_1)
    in_2 = InstanceNormalization()(c_2)
    lr_2 = LeakyReLU(alpha=0.01)(in_2)
    return lr_2


def decoder(pl, num_fil, has_skip_connection=False, connection=None, has_output=False):
    """
    Decoder function that applies Conv3DTranspose and Concatenation with encoder block convolutional layers
    :param pl: previous layer
    :param num_fil: number of filters in conv3d and conv3dTranspose layer
    :param has_skip_connection: whether has concatenation layer or not
    :param connection: encoder convolutional block return value to concatenate
    :param has_output: whether has output channels with sigmoid activation or not
    :return: tensor
    """
    ct = Conv3DTranspose(filters=num_fil, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(pl)
    if has_skip_connection is True and connection is not None:
        con = Concatenate()([ct, connection])
        ct = con
    c = _decoder(ct, num_fil)
    if has_output:
        out = Conv3D(filters=num_fil, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same", activation=sigmoid)(c)
        c = out
    return c


# Defining input layer
in_layer = Input(shape=(5, 128, 128, 128))
