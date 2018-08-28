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