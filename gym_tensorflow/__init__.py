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
from .tf_env import GymEnv, GymEnvWrapper, GymAtariWrapper
from.tf_subproc import SubprocVecEnv
from.import atari

def make(game, batch_size, *args, **kwargs):
    if game in atari.games:
        return atari.AtariEnv(game, batch_size, *args, **kwargs)
    if game == 'humanoid':
        return mujoco.HumanoidMO('Humanoid-v1', batch_size, *args, **kwargs)
    if game.startswith('par.'):
        return SubprocVecEnv([lambda: make(game[4:], 1, *args, **kwargs) for _ in range(batch_size)])
    if game.startswith('gym.'):
        return GymEnv(game[4:], batch_size, *args, **kwargs)
    raise NotImplementedError(game)


def get_ref_batch(make_env_f, sess, batch_size):
    env = make_env_f(1)
    assert env.discrete_action
    actions = tf.random_uniform((1,), minval=0, maxval=env.action_space, dtype=tf.int32)

    reset_op = env.reset()
    obs_op = env.observation()
    rew_op, done_op=env.step(actions)

    sess.run(tf.global_variables_initializer())

    sess.run(reset_op)

    ref_batch = []
    while len(ref_batch) < batch_size:
        obs, done = sess.run([obs_op, done_op])
        ref_batch.append(obs)
        if done.any():
            sess.run(reset_op)

    return np.concatenate(ref_batch)
