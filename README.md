## Replication of Ape-X (Distributed Prioritized Experience Replay)

This repo replicates the results Horgan et al obtained:

[1] [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)

Our code is based off of code from OpenAI baselines. The original code and related paper from OpenAI can be found [here](https://github.com/openai/baselines). Their implementation of DQN was modified to use Tensorflow custom ops.

Although Ape-X was originally a distributed algorithm, this implementation was meant to maximize throughput on a single machine. It was optimized for 2 GPUs (data gathering + optimization) but could be modified to use only one. With 2 GPUs and 20~40 CPUs you should be able to achieve human median performance in about 2 hours.

## How to run

clone repo

```
git clone https://github.com/uber-research/ape-x.git
```

create python3 virtual env

```
python3 -m venv env
. env/bin/activate
```

install requirements
```
pip install tensorflow-gpu gym
```

Follow the setup under `gym_tensorflow/README.md` and run `./make` to compile the custom ops.

launch experiment
```
python apex.py --env video_pinball --num-timesteps 1000000000 --logdir=/tmp/agent
```

Monitor your results with tensorboard
```
tensorboard --logdir=/tmp/agent
```

visualize results
```
python demo.py --env video_pinball --logdir=/tmp/agent
```

