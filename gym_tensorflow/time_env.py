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
import time
import threading

import tensorflow as tf

import gym_tensorflow
from gym_tensorflow.wrappers import AutoResetWrapper

def main(num_actors=128, num_threads=16):
    counter = tf.Variable(0, tf.int64)

    def make_env():
        env = AutoResetWrapper(gym_tensorflow.atari.AtariEnv('pong', num_actors))
        tf_rew, tf_done=env.step(tf.zeros((num_actors,), tf.int32))
        with tf.control_dependencies([tf_rew]):
            return tf.assign_add(counter, num_actors, use_locking=True)

    step_op = [make_env() for _ in range(num_threads)]

    def thread_f(sess, op):
        while True:
            sess.run(op)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        threads = [threading.Thread(target=thread_f, args=(sess, op)) for op in step_op]
        for t in threads:
            t.setDaemon(True)
            t._state = 0
            t.start()

        tstart = time.time()
        num_steps = 0
        while True:
            diff = sess.run(counter)-num_steps
            print('Rate: {:.0f} steps/s'.format(diff / (time.time() - tstart)))
            tstart = time.time()
            num_steps += diff
            time.sleep(5)

if __name__ == "__main__":
    main()