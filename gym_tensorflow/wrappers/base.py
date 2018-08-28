from gym_tensorflow.tf_env import TensorFlowEnv

class BaseWrapper(TensorFlowEnv):
    def __init__(self, env):
        self.env = env

    @property
    def batch_size(self):
        return self.env.batch_size

    @property
    def env_default_timestep_cutoff(self):
        return self.env.env_default_timestep_cutoff

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def discrete_action(self):
        return self.env.discrete_action

    def step(self, action, indices=None, name=None):
        return self.env.step(action=action, indices=indices, name=name)

    def reset(self, indices=None, max_frames=None, name=None):
        return self.env.reset(indices=indices, max_frames=max_frames, name=name)

    def observation(self, indices=None, name=None):
        return self.env.observation(indices=indices, name=name)

    def final_state(self, indices, name=None):
        return self.env.final_state(indices, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        return self.env.close()
