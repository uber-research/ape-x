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

import numpy as np
from multiprocessing import Process, Pipe

from .tf_env import PythonEnv

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    cum_reward = 0
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            reward, done = env._step([data], [0])
            remote.send((reward, done))
        elif cmd == 'reset':
            env._reset([0])
            cum_reward = 0
            remote.send(None)
        elif cmd == 'obs':
            ob = env._obs([0])
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(PythonEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            #p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    @property
    def discrete_action(self):
        return False

    def _step(self, actions, indices):
        for action, idx in zip(actions, indices):
            self.remotes[idx].send(('step', action))
        results = [self.remotes[idx].recv() for idx in indices]
        reward, done = zip(*results)
        return np.concatenate(reward), np.concatenate(done)

    def _reset(self, indices):
        for idx in indices:
            self.remotes[idx].send(('reset', None))
        for idx in indices:
            self.remotes[idx].recv()
        return 0

    def _obs(self, indices):
        for idx in indices:
            self.remotes[idx].send(('obs', None))
        results = [self.remotes[idx].recv() for idx in indices]
        return np.concatenate(results)

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)
