import os
import tensorflow as tf
import numpy as np

from gym_tensorflow.wrappers.base import BaseWrapper
from replay_buffer_actor import make_masked_frame
from ops.segment_tree import tfShortTermBuffer

class StackFramesWrapper(BaseWrapper):
    def __init__(self, env, num_stacked_frames=4, data_format='NCHW'):
        super(StackFramesWrapper, self).__init__(env)
        self.num_stacked_frames = num_stacked_frames
        self.data_format = data_format

        with tf.device('/cpu:0'):
            if data_format == 'NCHW':
                obs_space = env.observation_space[0], env.observation_space[-1], env.observation_space[1], env.observation_space[2]
            else:
                obs_space = env.observation_space
            self.buffer = tfShortTermBuffer(shapes=[obs_space, (env.batch_size,)], dtypes=[tf.uint8, tf.bool], framestack=num_stacked_frames, multi_step=0)

    @property
    def observation_space(self):
        return self.env.observation_space[:-1] + (self.env.observation_space[-1] * self.num_stacked_frames, )

    def observation(self, indices=None, reset=False, name=None):
        assert indices is None
        obs = self.env.observation(indices)
        if self.data_format == 'NCHW':
            obs = tf.transpose(obs, (0, 3, 1, 2))

        with tf.device('/cpu:0'):
            _, recent_obs_done = self.buffer.encode_history()

            observations, dones=zip( * recent_obs_done[1 - self.num_stacked_frames:])
            observations += (obs,)
            dones += (None,)

        return make_masked_frame(observations, dones, self.data_format)

    def step(self, action, indices=None, name=None):
        assert indices is None
        sliced_act_obs = self.env.observation(indices)
        if self.data_format == 'NCHW':
            sliced_act_obs = tf.transpose(sliced_act_obs, (0, 3, 1, 2))

        sliced_act_obs = tf.image.convert_image_dtype(sliced_act_obs, tf.uint8)
        assert sliced_act_obs.dtype == tf.uint8

        with tf.device('/cpu:0'):
            _, recent_obs_done = self.buffer.encode_history()

            observations, dones=zip( * recent_obs_done[1 - self.num_stacked_frames:])
            observations += (sliced_act_obs,)
            dones += (None,)

        obs = make_masked_frame(observations, dones, self.data_format)
        with tf.control_dependencies([sliced_act_obs]):
            rew, done = self.env.step(action=action, indices=indices, name=name)
            update_recent_history = self.buffer.enqueue([sliced_act_obs, done])

            with tf.control_dependencies([update_recent_history[0].op]):
                return tf.identity(rew), tf.identity(done)