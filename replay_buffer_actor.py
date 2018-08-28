'''
Copyright (c) 2018 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import tensorflow as tf
import numpy as np

import models
from ops.segment_tree import ShortTermBuffer

from gym_tensorflow.wrappers.base import BaseWrapper

def make_masked_frame(frames, dones, data_format):
    frames = list(frames[:])
    mask = None
    not_dones = [tf.cast(tf.logical_not(d), frames[0].dtype) if d is not None else None for d in dones]
    not_dones = [tf.expand_dims(d, axis=-1) if d is not None else None  for d in not_dones]
    not_dones = [tf.expand_dims(d, axis=-1) if d is not None else None  for d in not_dones]
    not_dones = [tf.expand_dims(d, axis=-1) if d is not None else None  for d in not_dones]
    for i in np.flip(np.arange(len(frames) - 1), 0):
        if mask is None:
            mask = not_dones[i]
        else:
            mask = mask * not_dones[i]
        frames[i] = tf.image.convert_image_dtype(frames[i] * mask, tf.float32)
    frames[-1] = tf.image.convert_image_dtype(frames[-1], tf.float32)
    if data_format == 'NHWC':
        return tf.concat(frames, axis=-1, name='concat_masked_frames')
    elif data_format == 'NCHW':
        return tf.concat(frames, axis=-3, name='concat_masked_frames')
    else:
        raise NotImplementedError()


class ReplayBufferWrapper(BaseWrapper):
    def __init__(self, env, actor_num, queue, num_stacked_frames, data_format):
        super(ReplayBufferWrapper, self).__init__(env)
        self.queue = queue
        self.actor_num = actor_num
        self.num_stacked_frames = num_stacked_frames
        self.data_format = data_format

        with tf.device('/cpu:0'):
            if data_format == 'NCHW':
                obs_space = env.observation_space[0], env.observation_space[-1], env.observation_space[1], env.observation_space[2]
            else:
                obs_space = env.observation_space
            self.buffer = ShortTermBuffer(shapes=[obs_space, (env.batch_size,)], dtypes=[tf.uint8, tf.bool], framestack=num_stacked_frames, multi_step=0)

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

            enqueue_op = self.queue.enqueue([obs, sliced_act_obs, rew, done, action, self.actor_num])

            with tf.control_dependencies([update_recent_history[0].op, enqueue_op]):
                return tf.identity(rew), tf.identity(done)


class PrioritizedReplayBufferWrapper(ReplayBufferWrapper):
    def __init__(self, *args, multi_step_n=None, **kwargs):
        super(PrioritizedReplayBufferWrapper, self).__init__(*args, **kwargs)
        self.transition_buffer = None
        self.multi_step_n = multi_step_n

    @classmethod
    def get_buffer_dtypes(cls, multi_step_n, framestack):
        return [tf.uint8, tf.float32, tf.bool, tf.int32, tf.float32, tf.float32] * (multi_step_n + framestack)

    @classmethod
    def get_buffer_shapes(cls, env, multi_step_n, num_stacked_frames, data_format):
        b = (env.batch_size,)
        if data_format == 'NCHW':
            obs_space = env.observation_space[-1], env.observation_space[1], env.observation_space[2]
        else:
            obs_space = env.observation_space[1:]
        shapes = [
            obs_space,  # Image
            (), # Reward
            (), # Done
            (), # Action
            (env.action_space,), # Q Values
            (), # Selected Q Value
        ]
        shapes = [b + s for s in shapes]
        return shapes * (multi_step_n + num_stacked_frames)

    def step(self, action, indices=None, name=None, q_values=None, q_t_selected=None):
        assert indices is None
        assert q_values is not None
        assert q_t_selected is not None
        batch_size = self.env.batch_size
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

            current_frame = sliced_act_obs, rew, done, action, q_values, q_t_selected
            if self.transition_buffer is None:
                with tf.control_dependencies(None):
                    with tf.device('/cpu:0'):
                        self.transition_buffer = ShortTermBuffer(shapes=[v.get_shape() for v in current_frame], dtypes=[v.dtype for v in current_frame], framestack=self.num_stacked_frames, multi_step=self.multi_step_n)
            is_valid, history = self.transition_buffer.enqueue(current_frame)

            history = [e for t in history for e in t]
            replay_queue_shapes = [(None,) + tuple(a.get_shape()[1:]) for a in history]

            enqueue_op = tf.cond(is_valid, lambda: self.queue.enqueue(history), tf.no_op)

            with tf.control_dependencies([enqueue_op, update_recent_history[0].op]):
                return tf.identity(rew), tf.identity(done)
