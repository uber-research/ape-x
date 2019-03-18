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
import tensorflow.contrib.slim
from tensorflow.contrib.slim import prefetch_queue
import numpy as np

import tf_util as U

import gym_tensorflow
from gym_tensorflow.wrappers import AutoResetWrapper

from replay_buffer_actor import PrioritizedReplayBufferWrapper, make_masked_frame

from ops.segment_tree import ReplayBuffer
import models


def main(env, num_timesteps=int(10e6), dueling=True, **kwargs):
    env_f = lambda batch_size: gym_tensorflow.make(env, batch_size=batch_size)
    model = models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=bool(dueling),
    )
    act = learn(
        env_f,
        q_func=model,
        max_timesteps=int(num_timesteps),
        dueling=True,
        **kwargs
    )


def build_act(actor_num, env, q_func, num_actions, eps, scope="deepq", data_format=None, reuse=None, replay_queue=None, prioritized_replay_eps=None, gamma=None, replay_queue_capacity=None, multi_step_n=1, framestack=4, num_actor_steps=None):
    # Build the actor using \epsilon-greedy exploration
    # Environments are wrapped with a data gathering class
    with tf.variable_scope('deepq', reuse=reuse):
        with tf.device('/gpu:0'):
            act_obs = env.observation()
            q_values = q_func(act_obs, num_actions, scope="read_q_func", data_format=data_format)
        deterministic_actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
        batch_size = deterministic_actions.get_shape()[0]
        with tf.device('/cpu:0'):
            random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int32)
            chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
            output_actions = tf.where(chose_random, random_actions, deterministic_actions)

            q_t_selected = tf.reduce_sum(q_values * tf.one_hot(output_actions, num_actions), 1)

        with tf.control_dependencies([tf.assign_add(num_actor_steps, batch_size, use_locking=True)]):
            return env.step(output_actions, q_values=q_values, q_t_selected=q_t_selected)


def build_train(train_dequeue, num_training_steps, q_func, num_actions, optimizer, grad_norm_clipping=None, data_format=None, gamma=None, multi_step_n=1,
    double_q=True, scope="deepq", reuse=None, replay_buffer=None, prioritized_replay_eps=None,
    bellman_h=None, bellman_ih=None,
    use_temporal_consistency=True):
    with tf.variable_scope(scope, reuse=reuse):
        actor_num, obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph, importance_weights_ph, idxs = train_dequeue

        # q network evaluation
        q_t = q_func(obs_t_input, num_actions, scope="q_func", data_format=data_format)
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input, num_actions, scope="target_q_func", data_format=data_format)
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input, num_actions, scope="q_func", reuse=True, data_format=data_format)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = tf.stop_gradient((1.0 - done_mask_ph) * q_tp1_best)

        # compute RHS of bellman equation
        q_t_selected_target = bellman_h(rew_t_ph + gamma ** multi_step_n * bellman_ih(q_tp1_best_masked))
        q_t_selected_target = tf.stop_gradient(q_t_selected_target)

        # compute the error (potentially clipped)
        td_error = q_t_selected - q_t_selected_target
        errors = U.huber_loss(td_error)

        # This TC component was used by Pohlen et. al. to allow higher discounting factors
        # It seems to slow down learning so I disabled for the demo, the authors claimed it improves asymptotic performance
        if use_temporal_consistency:
            q_tp1_best_using_online_net_masked = (1.0 - done_mask_ph) * tf.reduce_max(q_tp1_using_online_net, 1)
            tc_error = q_tp1_best_using_online_net_masked - q_tp1_best_masked
            errors = errors + U.huber_loss(tc_error)

        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group( * update_target_expr)

        # To avoid unnecessary copies between gpus we maintain a copy on actors GPU that is updated each iteration
        with tf.device('/gpu:0'):
            q_func(obs_t_input, num_actions, scope="read_q_func", data_format=data_format, reuse=True)
            read_q_func_vars = U.scope_vars(U.absolute_scope_name("read_q_func"))
        update_read_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(read_q_func_vars, key=lambda v: v.name)):
            update_read_expr.append(var_target.assign(var))
        update_read_expr = tf.group( * update_read_expr)

        if replay_buffer:
            new_priorities = tf.abs(td_error) + prioritized_replay_eps
            update_priority = replay_buffer.assign(idxs, new_priorities)
            optimize_expr = tf.group([optimize_expr, update_priority])

        with tf.control_dependencies([optimize_expr, update_read_expr]):
            train = tf.assign_add(num_training_steps, 1)

        return train, update_target_expr


def learn(env_f,
          q_func,
          max_timesteps=100000,
          buffer_size=2 ** 21,
          num_actors=4,
          actor_batch_size=384,
          batch_size=512,
          print_freq=1,
          multi_step_n=3,
          learning_starts=50000,
          gamma=0.99,
          grad_norm_clipping=40,
          target_network_update_freq=2500,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_eps=1e-6,
          number_of_prefetched_batches=16,
          number_of_prefetching_threads=4,
          number_of_actor_buffer_threads=3,
          actor_buffer_capacity=16,
          framestack=4,
          data_format='NCHW',
          use_transformed_bellman=False,
          use_temporal_consistency=False,
          logdir='/tmp/agent',
          optimizer={"type": "adam", "args": {"learning_rate": 0.00025 / 4}},
          #optimizer={"type": "rmsprop", "args": {"learning_rate": 0.00025 / 4, "decay": 0.95, "epsilon": 1.5e-7, "momentum": 0, "centered": True}},
          **kwargs):
    tf.logging.set_verbosity(tf.logging.INFO)
    print(locals())
    bellman_eps = 1e-2
    # Create all the functions necessary to train the model
    if use_transformed_bellman:
        bellman_h = lambda x: tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + x * bellman_eps
        bellman_ih = lambda x: tf.sign(x) * ((tf.sqrt(1 + 4 * bellman_eps * (tf.abs(x) + 1 + bellman_eps)) - 1)** 2 * (1 / (2 * bellman_eps))** 2 - 1)
        reward_clipping = lambda x: x
    else:
        reward_clipping = lambda x: tf.clip_by_value(x, -1, 1)
        bellman_h = lambda x:x
        bellman_ih = bellman_h

    ####################################################################
    # Make Actors

    assert actor_batch_size % num_actors == 0
    envs = [AutoResetWrapper(env_f(actor_batch_size // num_actors), max_frames=50000) for actor_num in range(num_actors)]

    actor_fifo = tf.FIFOQueue(actor_buffer_capacity, dtypes=PrioritizedReplayBufferWrapper.get_buffer_dtypes(multi_step_n, framestack), shapes=PrioritizedReplayBufferWrapper.get_buffer_shapes(envs[0], multi_step_n, framestack, data_format=data_format))

    tf.summary.scalar("actor_buffer/fraction_of_%d_full" % actor_buffer_capacity,
                      tf.to_float(actor_fifo.size()) * (1. / actor_buffer_capacity))
    envs = [PrioritizedReplayBufferWrapper(envs[actor_num], actor_num, actor_fifo, framestack, data_format, multi_step_n=multi_step_n) for actor_num in range(num_actors)]

    # Compute \epsilon for each actor
    alpha = 7
    base_eps = 0.4
    eps_array = base_eps ** (1 + np.flip(np.arange(actor_batch_size), 0) * alpha / (actor_batch_size - 1))
    eps_array = np.reshape(eps_array, [num_actors, -1])

    act = []
    num_actor_steps = tf.get_variable('num_actor_steps', shape=(), dtype=tf.int64, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    def make_actors():
        for actor_num, (env, eps) in enumerate(zip(envs, eps_array)):
            act_f = build_act(actor_num, env, q_func, env.action_space, eps,
                                                                    scope='deepq_actor',
                                                                    data_format=data_format,
                                                                    reuse=actor_num > 0,
                                                                    replay_queue=actor_fifo, replay_queue_capacity=actor_buffer_capacity,
                                                                    prioritized_replay_eps=prioritized_replay_eps,
                                                                    gamma=gamma,
                                                                    num_actor_steps=num_actor_steps,
                                                                    multi_step_n=multi_step_n, framestack=framestack)
            act.append(act_f)
    make_actors()

    # This class reports episode rewards for actor_0 (lowest \epsilon/most greedy)
    actor_monitor = ActorMonitor(logdir=logdir)
    rew, done = act[0]
    act[0] = actor_monitor(rew, done, num_actor_steps)

    # Make actor threads
    qr = tf.train.QueueRunner(actor_fifo, [tf.group(a) for a in act])
    tf.train.add_queue_runner(qr)

    ####################################################################
    # Make transitions from actor fifo
    # This essentially computes the priority score as well as compute the n-step reward

    def make_transition_from_history(history):
        # sliced_act_obs, rew, done, output_actions, q_values, q_t_selected
        assert len(history) == 6 * (multi_step_n + framestack)
        history = [history[i:i + 6] for i in range(0, len(history), 6)]

        old = history[-1 - multi_step_n]
        new = history[-1]

        q_values = new[4]

        observations = list(list(zip( * history))[0])
        rewards = list(list(zip( * history))[1])
        dones = list(list(zip( * history))[2])

        def make_masked_reward(rewards, dones):
            rewards = list(rewards[:])
            zeros = tf.zeros_like(rewards[0])
            mask = dones[0]
            for i in np.arange(1, len(rewards)):
                rewards[i] = tf.where(mask, zeros, rewards[i])
                mask = tf.logical_or(mask, dones[i])
            return rewards, mask

        rewards, end_of_episode = make_masked_reward(rewards[ -1- multi_step_n: - 1], dones[ -1- multi_step_n: - 1])
        assert len(rewards) == multi_step_n
        discounted_reward = sum([reward_clipping(rew) * gamma ** n for n, rew in enumerate(rewards)])

        _, _, _, old_action, _, old_q_values_selected=old
        end_of_episode = tf.cast(end_of_episode, dtype=tf.float32)

        discounted_q = bellman_h(discounted_reward + gamma ** multi_step_n * (1.0 - end_of_episode) * bellman_ih(tf.reduce_max(q_values, axis=1)))
        td_err = old_q_values_selected - discounted_q
        priority = tf.abs(td_err) + prioritized_replay_eps

        return [tf.zeros_like(old_action), old_action, discounted_reward, end_of_episode] + observations + dones, priority
    actor_data, priority = make_transition_from_history(actor_fifo.dequeue())

    ####################################################################
    # Make Actors post-processing and Replay buffers
    # This buffers are custom tf ops to maximize throughput without requiring multiple copies of each frame
    # 2M frames are about 13GB of data

    # Create the replay buffer
    replay_buffer = ReplayBuffer(buffer_size,
                                   shapes=[d.get_shape()[1:] for d in actor_data],
                                   dtypes=[d.dtype for d in actor_data],
                                   alpha=prioritized_replay_alpha)
    tf.summary.scalar("replay_buffer/fraction_of_%d_full" % buffer_size,
                      tf.to_float(replay_buffer.size()) * (1. / buffer_size))

    update_replay_buffer_op = replay_buffer.enqueue_many(actor_data, priorities=priority)

    qr = tf.train.QueueRunner(actor_fifo, [update_replay_buffer_op] * number_of_actor_buffer_threads, close_op=replay_buffer.close())
    tf.train.add_queue_runner(qr)

    ####################################################################
    # Make Learner input pipeline

    num_training_steps = tf.get_variable('training_steps', shape=(), dtype=tf.int64)
    prioritized_replay_beta = U.linear_schedule(
            num_training_steps, num_timesteps, prioritized_replay_beta0, 1.0)

    def make_training_input():
        with tf.variable_scope("training_input_preprocessing"):
            transition = replay_buffer.sample_proportional_from_buffer(batch_size, prioritized_replay_beta, minimum_sample_size=learning_starts)

            # GPU because in our SKU the CPUs were the bottleneck
            with tf.device('/gpu:1'):
                idxes, weights, actor_num, transition_action, transition_reward, transition_done=transition[:6]
                frames = transition[6:]
                assert len(frames) == (framestack + multi_step_n) * 2

                # Handle edge cases (done = True)
                frames, dones=frames[:framestack + multi_step_n], frames[framestack + multi_step_n:]
                obs_t = make_masked_frame(frames[:framestack], dones[:framestack], data_format)
                obs_tp1 = make_masked_frame(frames[ - framestack:], dones[ - framestack:], data_format)

                return actor_num, obs_t, transition_action, transition_reward, obs_tp1, transition_done, weights, idxes

    with tf.variable_scope("training_queue"):
        training_fifo = prefetch_queue.prefetch_queue(make_training_input(), capacity=number_of_prefetched_batches, num_threads=number_of_prefetching_threads)
        train_queue_size = training_fifo.size()

    ####################################################################
    # Make Learner

    optimizer = {'adam': tf.train.AdamOptimizer, 'rmsprop': tf.train.RMSPropOptimizer}[optimizer['type']](**optimizer['args'])

    tf.summary.scalar('num_training_steps', num_training_steps)
    tf.summary.scalar('training_transitions', batch_size * num_training_steps)

    train_dequeue = training_fifo.dequeue()
    with tf.device('/gpu:1'):
        # Prefetch data into the GPU every iteration
        staging_area = tf.contrib.staging.StagingArea(
                    [t.dtype for t in train_dequeue],
                    [t.shape for t in train_dequeue],
                    capacity=1)
        stage_op = staging_area.put(train_dequeue)

        train, update_target = build_train(
            train_dequeue=staging_area.get(),
            num_training_steps=num_training_steps,
            q_func=q_func,
            data_format=data_format,
            num_actions = envs[0].action_space,
            gamma=gamma, multi_step_n=multi_step_n,
            optimizer=optimizer,
            grad_norm_clipping=grad_norm_clipping,
            replay_buffer=replay_buffer,
            prioritized_replay_eps=prioritized_replay_eps,
            use_temporal_consistency=use_temporal_consistency,
            bellman_h=bellman_h, bellman_ih=bellman_ih,
        )

    ####################################################################
    # Finally, we get to start it

    summary_writer = tf.summary.FileWriterCache.get(logdir)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    scaffold = tf.train.Scaffold()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=600,
        save_summaries_secs=None,
        save_summaries_steps=None,
        log_step_count_steps=None,
        scaffold=scaffold,
        config=config,
        hooks=[tf.train.StepCounterHook(every_n_steps=None, every_n_secs=10, summary_writer=summary_writer),
              tf.train.SummarySaverHook(save_secs=30, scaffold=scaffold, summary_writer=summary_writer),
              tf.train.StopAtStepHook(last_step=max_timesteps)]) as sess:
        sess.run_step_fn(
            lambda step_context: step_context.session.run([update_target, stage_op]))
        tf.logging.info('Training started')
        while not sess.should_stop():
            total_steps, _ = sess.run([train, stage_op])

            if total_steps % target_network_update_freq == 0:
                # Update target network periodically.
                sess.run_step_fn(lambda step_context: step_context.session.run([update_target]))


class ActorMonitor(object):
    ####################################################################
    # Need a way to monitor rewards for actor_0
    # This is hacky but does the trick
    def __init__(self, logdir):
        self.episode_rewards = [0.0]
        self.episode_length = [0]
        self.num_episodes = 0
        self.summary_writer = tf.summary.FileWriterCache.get(logdir)

    def __call__(self, rew, done, num_env_frames):
        agent_0_steps = tf.get_variable('agent_0_steps', shape=(), dtype=tf.int64)
        tf.summary.scalar('agent_0/total_steps', agent_0_steps)
        with tf.control_dependencies([tf.assign_add(agent_0_steps, 1)]):
            return [tf.py_func(self.report_actor, (rew, done, num_env_frames,), tf.float64, stateful=True, name='ActorSummaryReport')]

    def report_actor(self, rew, done, num_env_frames):
        self.episode_rewards[-1] += rew[0]
        self.episode_length[-1] += 1
        if done[0]:
            tf.logging.info('Episode done: {}'.format(self.episode_rewards[-1]))
            self.episode_rewards.append(0.0)
            self.episode_rewards = self.episode_rewards[-101:]
            self.episode_length.append(0)
            self.episode_length = self.episode_length[-101:]
            self.num_episodes += 1

            summary = tf.summary.Summary()
            summary.value.add(tag='agent_0/completed_episodes', simple_value=self.num_episodes)
            summary.value.add(tag='agent_0/last_episode_reward', simple_value=np.mean(self.episode_rewards[-2]))
            summary.value.add(tag='agent_0/mean_100_episode_reward', simple_value=np.mean(self.episode_rewards[:-1]))
            summary.value.add(tag='agent_0/median_100_episode_reward', simple_value=np.median(self.episode_rewards[:-1]))
            summary.value.add(tag='agent_0/max_100_episode_reward', simple_value=np.max(self.episode_rewards[:-1]))
            summary.value.add(tag='agent_0/mean_100_episode_length', simple_value=np.mean(self.episode_length[:-1]))
            self.summary_writer.add_summary(summary, num_env_frames)
        return 0.0


def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='breakout')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--logdir', default='/tmp/apex')
    args = parser.parse_args()
    main(env=args.env, num_timesteps=args.num_timesteps)


if __name__ == '__main__':
    cli()
