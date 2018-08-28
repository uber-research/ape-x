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
import os
import time
import argparse

import tensorflow as tf
import numpy as np

from ailabs import tlogger as logger
import gym

import gym_tensorflow
import gym_tensorflow.wrappers
import models
from atari_wrappers import FireResetEnv, NoopResetEnv, MaxAndSkipEnv

from stack_frames import StackFramesWrapper

def main(env, dueling=True, **kwargs):
    env = FireResetEnv(MaxAndSkipEnv(NoopResetEnv(gym.make(env))))
    # Or equivalent using gym_tensorflow
    #env = gym_tensorflow.make(env, 1)
    model = models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=bool(dueling),
    )
    act = demo(
        env,
        q_func=model,
        dueling=True,
        **kwargs
    )


def demo(gym_env, q_func, dueling, logdir, data_format='NCHW', num_episodes=200, max_frames=50000):
    tf.logging.set_verbosity(tf.logging.INFO)
    env = gym_tensorflow.GymAtariWrapper([gym_env])
    env = gym_tensorflow.wrappers.ImageRescaleWrapper(env)
    env = StackFramesWrapper(env, data_format=data_format)

    num_actions = env.action_space

    reset_op = env.reset()

    # Build the actor using \epsilon-greedy exploration
    # Environments are wrapped with a data gathering class
    with tf.variable_scope('deepq'):
        act_obs = env.observation()
        q_values = q_func(act_obs, num_actions, scope="q_func", data_format=data_format)
        deterministic_actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
        rew_op, done_op = env.step(deterministic_actions)

    ####################################################################
    with tf.train.SingularMonitoredSession(
        checkpoint_dir=logdir,) as sess:

        ep_rew = []
        for _ in range(num_episodes):
            cumrew = 0.0
            sess.run(reset_op)
            gym_env.render()
            for i in range(max_frames):
                rew, done = sess.run([rew_op, done_op])
                gym_env.render()
                cumrew += rew
                if done:
                    break
            print('Episode finished: {:.0f} @ {} frames'.format(cumrew[0], i))
            ep_rew.append(cumrew)
        print('Mean reward:', np.mean(ep_rew))


def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='breakout')
    parser.add_argument('--logdir', default='/tmp/apex')
    args = parser.parse_args()
    main(env=args.env, logdir=args.logdir)


if __name__ == '__main__':
    cli()