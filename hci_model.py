"""
    Hard Coded Implementation of Optimized U-Net
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""


from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, Concatenate, LeakyReLU, MaxPooling3D, Input)
