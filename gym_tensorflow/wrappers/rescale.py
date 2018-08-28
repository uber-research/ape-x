import os
import tensorflow as tf
import numpy as np

from .base import BaseWrapper

class ImageRescaleWrapper(BaseWrapper):
    def __init__(self, env, warp_size=(84, 84)):
        super(ImageRescaleWrapper, self).__init__(env)
        self.warp_size = warp_size

    @property
    def observation_space(self):
        return (self.batch_size,) + self.warp_size + self.env.observation_space[-1:]

    def observation(self, *args, **kwargs):
        '''Returns current observation after preprocessing (skip, grayscale, warp, stack).\nMust be called ONCE each time step is called if num_stacked_frames > 1
        '''
        obs = self.env.observation(*args, **kwargs)
        obs = tf.image.resize_bilinear(obs, self.warp_size, align_corners=True)
        obs = tf.reshape(obs, (self.batch_size,) + self.warp_size + (1,))
        return obs
