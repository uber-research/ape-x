import os
import tensorflow as tf
import numpy as np

from .base import BaseWrapper

class AutoResetWrapper(BaseWrapper):
    def __init__(self, env, max_frames=None):
        super(AutoResetWrapper, self).__init__(env)
        self.max_frames = max_frames

    def step(self, action, indices=None, name=None):
        rew, done = self.env.step(action=action, indices=indices, name=name)
        if indices is None:
            indices = np.arange(self.batch_size, dtype=np.int32)
        done_idxs = tf.boolean_mask(indices, done)
        with tf.control_dependencies([self.reset(done_idxs, max_frames=self.max_frames)]):
            return tf.identity(rew), tf.identity(done)
